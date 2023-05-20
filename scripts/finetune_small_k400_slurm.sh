export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# Set the path to save checkpoints
OUTPUT_DIR=''
# path to annotation file (train.csv/val.csv/test.csv)
DATA_PATH=''
# path to pretrain model
MODEL_PATH=''

JOB_NAME=$1
PARTITION=${PARTITION:-"3dv-share"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --data_set Kinetics-400 \
        --nb_classes 400 \
        --batch_size 32 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --opt adamw \
        --lr 1e-3 \
        --warmup_lr 1e-6\
        --smoothing 0.1\
        --mixup 0.8\
        --cutmix 1.0\
        --min_lr 1e-5 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 150 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --fc_drop_rate 0.0 \
        --drop_path 0.1 \
        --use_checkpoint \
        --dist_eval \
        --enable_deepspeed \
        ${PY_ARGS}
