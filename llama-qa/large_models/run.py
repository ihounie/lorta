import argparse
from transformers import TrainerCallback, Trainer, AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from tqdm import tqdm
from tasks import get_task
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
import random
import wandb
from transformers.models.llama.modeling_llama import LlamaRMSNorm


from transformers import Trainer
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    overwrite_output_dir: bool = True
    output_dir: str = './trained_models/llama-tt'
    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = True # load model parameters as bfloat16
    bf16 = True
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    # options
    # - none: no training -- for zero-shot or in-context learning (ICL)
    # - regular: regular huggingface trainer
    # - gpt: test the zero-shot pefromance with OpenAI API
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # parameter setup for PEFT methods
    # LoRETTA
    tensor_rank: int = 8
    target_modules: List[str] = None # set to be None when use official support model
    task_type: str = 'CAUSAL_LM' # choose from "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"
    # LoRETTA_adp
    adp_bottleneck: int = 64
    non_linearity: str = "relu"
    adapter_dropout: float = 0.0
    scaling: Union[float, str] = 1.0
    # LoRETTA_rep
    rep_bottleneck: int = 8
    rep_alpha: int = 16
    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words
    rank: int = 8
    # LoRA
    tuning_type: str = 'ft'
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)

    remove_unused_columns: bool=False 

def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args

def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters()) / 1000 / 1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000 / 1000
    wandb.log({"Total(M)": total_num, "Trainable(M)": trainable_num})
    return {'Total(M)': total_num, 'Total Trainable(M)': trainable_num}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    low_cpu_mem_usage=True,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    load_in_8bit=self.args.load_int8,
                    low_cpu_mem_usage=True,
                )
            model.eval()

            # Check for NaNs
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    raise ValueError(f"NaN values found in parameter {name} immediately after loading")
            

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if "llama" in self.args.model_name:
            # LLaMA padding token
            tokenizer.pad_token_id = 0 # technically <unk>

        # Prefix tuning/LoRA
        if self.args.tuning_type == 'prefix':
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.tuning_type == 'lora':
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=None,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        if self.args.tuning_type == 'lorta':
            from peft import LorTaConfig, get_peft_model
            config = LorTaConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        if self.args.tuning_type == 'loretta_rep':
            from loretta import LorettaRepConfig, get_peft_model
            config = LorettaRepConfig(
                r=self.args.rep_bottleneck,
                lora_alpha=self.args.rep_alpha,
                target_modules=self.args.target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=self.args.task_type,
                tensor_rank=self.args.tensor_rank
            )
            model = get_peft_model(model, config)

            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (LlamaRMSNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        if self.args.tuning_type == 'adapters':
            from peft_local import BottleneckConfig, get_peft_model  # noqa: E402
            bottleneck_size: int = 64
            non_linearity: str = "relu"
            adapter_dropout: float = 0.0
            use_parallel_adapter: bool = False
            use_adapterp: bool = False
            target_modules: List[str] = None
            scaling: Union[float, str] = 1.0
            config = BottleneckConfig(
                bottleneck_size=bottleneck_size,
                non_linearity=non_linearity,
                adapter_dropout=adapter_dropout,
                use_parallel_adapter=use_parallel_adapter,
                use_adapterp=use_adapterp,
                target_modules=target_modules,
                scaling=scaling,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (LlamaRMSNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        if self.args.tuning_type == 'prompt':
            # from prefix import PrefixTuning
            # PrefixTuning(model, num_prefix=5, reparam=False, float16=False, init_by_real_act=True)
            from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=8,
                prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
                tokenizer_name_or_path=self.args.model_name,
            )
            model = get_peft_model(model, peft_config)
        if self.args.tuning_type == 'ia3':
            # from prefix import PrefixTuning
            # PrefixTuning(model, num_prefix=5, reparam=False, float16=False, init_by_real_act=True)
            from peft import get_peft_model, IA3Config, TaskType
            peft_config = IA3Config(
                task_type=TaskType.SEQ_CLS, target_modules=["k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"]
            )

            model = get_peft_model(model, peft_config)
        if self.args.tuning_type == 'loretta_adp':
            from loretta import LorettaAdpConfig, get_peft_model
            peft_config = LorettaAdpConfig(
                bottleneck_size=self.args.adp_bottleneck,
                non_linearity=self.args.non_linearity,
                adapter_dropout=self.args.adapter_dropout,
                target_modules=self.args.target_modules,
                scaling=self.args.scaling,
                bias="none",
                task_type=self.args.task_type,
                tensor_rank=self.args.tensor_rank,
            )
            model = get_peft_model(model, peft_config)
        # print the name and shape of trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape} parameters")
        logger.info("Total Parameter Count: {}M".format(model.num_parameters() / 1000 / 1000))
        logger.info("Total and trainable params: {}".format(str(get_parameter_number(model))))
        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            # kwargs = {}
            outputs = self.model.generate(
                input_ids=input_ids, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")
        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            # print(f'Evaluating {eval_id} Eval_sample {eval_sample}')
            # print(f'prediction {self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))}')
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification


        def build_compute_metrics_fn(task_name, train_samples, eval_samples, one_train_set_per_eval_sample=False):
            def compute_metrics_fn(p):
                predictions = []
                for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
                    predictions.append(
                        self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                           eval_sample, verbose=(eval_id < 3))
                    )
                # Calculate metrics
                metric_name = getattr(task_name, "metric_name", "accuracy")
                metrics = {metric_name: calculate_metric(predictions, metric_name)}
                return metrics
            return compute_metrics_fn

        class AdjustLRCallback(TrainerCallback):
            def __init__(self, step_to_adjust, new_lr):
                self.step_to_adjust = step_to_adjust
                self.new_lr = new_lr

            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step == self.step_to_adjust:
                    print(f'change learning rate to {self.new_lr}')
                    for param_group in kwargs['optimizer'].param_groups:
                        param_group['lr'] = self.new_lr
                    kwargs['lr_scheduler'].base_lrs = [self.new_lr for _ in kwargs['lr_scheduler'].base_lrs]

        class LogLinearLayerWeightsCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, model=None, **kwargs):
                """
                Called at the beginning of an epoch during training.
                :param args: Training arguments.
                :param state: Training state.
                :param control: Training control.
                :param model: The model being trained.
                """
                # Ensure the model is provided and in evaluation mode for safe weight access
                if model is not None:
                    model.eval()

                    # Iterate through the model's named parameters to find the layer of interest
                    for idx, param in enumerate(model.model.model.layers[1].mlp.up_proj.adapter_down.layer.tensor.factors):
                        weight_mean = torch.mean(param.data).item()
                        wandb.log({f"mean_{idx}": weight_mean, "epoch": state.epoch})
                        model.train()
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()
    args.lora_r = args.rank
    set_seed(args.seed)
    task = get_task(args.task_name)
    print(f'check {args.num_dev}')
    if args.num_train>0:
        train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
    else:
        train_sets = [task.samples["train"]]
    wandb_run_name = str(args.task_name) + '-' + str(args.model_name.replace('/', '-')) + '-' \
                     + str(args.learning_rate) + '-' \
                     + str(args.tuning_type.replace('_', '-')) + '-rank-' + str(args.lora_r)
    wandb.init(project="loretta-llama", name=wandb_run_name)
    # Initialize trainer and load model
    if args.trainer == 'gpt':
        # zero-shot training with openai API
        print(f'Prediction with openai API')
        import openai
        import csv
        from message import build_message
        test_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)
        for eval_id, eval_sample in enumerate(tqdm(test_samples)):
            print(f'Evaluating {eval_id} Eval_sample {eval_sample}')
        OPENAI_API_KEY = ""
        openai.api_key = OPENAI_API_KEY

        def get_completion_from_messages(messages, model="gpt-4-1106-preview",
                                         temperature=0, max_tokens=500):
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message["content"]

        test_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)
        # Prediction loop
        predictions = []
        labels = []
        for eval_id, eval_sample in enumerate(tqdm(test_samples)):
            messages, label = build_message(args.task_name, eval_sample)
            response = get_completion_from_messages(messages)
            predictions.append(response)
            labels.append(label)
            # print(f'{response},{label}')
        with open(f'output_{args.task_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Prediction', 'Label'])
            for prediction, label in zip(predictions, labels):
                writer.writerow([prediction, label])
        # metric_name = getattr(self.task, "metric_name", "accuracy")
        # metrics = {metric_name: calculate_metric(predictions, metric_name)}
        # return metrics
    else:
        # training with huggingface trainer
        framework = Framework(args, task)
        if args.train_set_seed is not None or args.num_train_sets is not None:
            # Eval samples share one (or multiple) training set(s)
            for train_set_id, train_samples in enumerate(train_sets):
                train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

                # Sample eval samples
                if args.num_eval is not None:
                    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                else:
                    eval_samples = task.valid_samples

                if args.trainer != "none":
                    if args.num_dev is not None:
                        # Dev samples
                        dev_samples = train_samples[-args.num_dev:]
                        train_samples = train_samples[:-args.num_dev]
                    else:
                        dev_samples = None

                    # Training
                    framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                    if not args.no_eval:
                        metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                        if dev_samples is not None:
                            dev_metrics = framework.evaluate([], dev_samples)
                            for m in dev_metrics:
                                metrics["dev_" + m] = dev_metrics[m]
                        for metric in metrics:  
                            wandb.log({metric: metrics[metric]})
                else:
                    print(f'check {args.num_dev}')
                    # assert args.num_dev is None
                    # Zero-shot / in-context learning
                    metrics = framework.evaluate(train_samples, eval_samples)

                if not args.no_eval:
                    logger.info("===== Train set %d =====" % train_set_seed)
                    logger.info(metrics)
                    if args.local_rank <= 0:
                        write_metrics_to_file(metrics, "./results/" + result_file_tag(args) + f"-trainset{train_set_id}-rank-{args.rank}" + ".json" if args.result_file is None else args.result_file)
        else:
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            metrics = framework.evaluate(train_sets, eval_samples, False)
            for metric in metrics:
                wandb.log({"eval/"+metric: metrics[metric]})


if __name__ == "__main__":
    main()
