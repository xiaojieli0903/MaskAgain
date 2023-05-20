OUTPUT_DIR=''
DATA_PATH=''



OMP_NUM_THREADS=1  python3 -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12322 ../run_mae_kd.py \
        --mask_type tube \
        --mask_ratio 0.75 \
        --model_s pretrain_videomae_base_patch16_224 \
        --decoder_depth 0 \
        --lr 3e-4 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 100 \
        --save_ckpt_freq 50 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}\
        --data_path ${DATA_PATH} \
        --distill hybrid\
        --kd_T 1\
        --beta 1\
        --beta_kd1 0.1\
        --beta_kd2 3\
        --gamma 0.1\
        --alpha 0.1\
        --temperature_s 0.1\
        --temperature_t 0.05\
        --num_workers 8\
        --num_k 10\
        --num_f 10\
        --path_t 