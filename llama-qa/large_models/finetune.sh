MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-5}
BS=${BS:-4}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
RANK=${RANK:-8}
MODE=${MODE:-ft}
DEVICE=${DEVICE:-0}
TRAINER=${TRAINER:-regular}
export CUDA_VISIBLE_DEVICES=$DEVICE
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TYPE="prefix"
elif [ "$MODE" == "lora" ]; then
    TYPE="lora"
elif [ "$MODE" == "loretta_rep" ]; then
    TYPE="loretta_rep"
elif [ "$MODE" == "adapters" ]; then
    TYPE="adapters"
elif [ "$MODE" == "loretta_adp" ]; then
    TYPE="loretta_adp"
elif [ "$MODE" == "prompt" ]; then
    TYPE="prompt"
elif [ "$MODE" == "bitfit" ]; then
    TYPE="bitfit"
elif [ "$MODE" == "ft" ]; then
    TYPE="ft"
elif [ "$MODE" == "ia3" ]; then
    TYPE="ia3"
elif [ "$MODE" == "lorta" ]; then
    TYPE="lorta"
fi
TAG=$MODE-$EPOCH-$BS-$LR-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        BS=1
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    SQuAD)
        BS=1
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "TYPE: $TYPE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

#python drun.py \
#    --model_name $MODEL \
#    --task_name $TASK \
#    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --tuning_type $TYPE --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
#    --trainer regular \
#    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
#    --load_best_model_at_end --evaluation_strategy epoch --save_strategy no --save_total_limit 1 \
#    --train_as_classification \
#    $EXTRA_ARGS \
#    $TASK_ARGS \
#    "$@"

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir ./results/$TASK-${MODEL_NAME}-$RANK-$TAG --tag $TAG --tuning_type $TYPE --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --trainer $TRAINER --rank $RANK \
    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
    --load_best_model_at_end --evaluation_strategy steps --eval_step 50 --save_strategy steps --save_step 50 --save_total_limit 1 \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
