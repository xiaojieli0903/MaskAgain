export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR=''
DATA_PATH='./train.csv'

JOB_NAME=$1
PARTITION=${PARTITION:-"3dv-share"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
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
        python -u run_kd.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_small_patch16_224 \
        --decoder_depth 0 \
        --batch_size 64 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --drop_path 0.1 \
        --weight_decay 0.05 \
        --epochs 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}\
        --lr 1.5e-4 \
        --distill hybrid\
        --kd_T 1\
        --num_k 15\
        --num_f 15\
        --beta 1\
        --beta_kd1 0.1\
        --beta_kd2 3\
        --gamma 0.1\
        --alpha 0.1\
        --num_workers 5\
        --path_t ../work_dirs/k400_base_1600e_checkpoint.pth \
        ${PY_ARGS}
