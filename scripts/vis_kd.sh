# path to video for visualization
VIDEO_PATH='./ucf101/videos/SoccerJuggling/v_SoccerJuggling_g04_c04.avi'



AREC='pretrain_videomae_base_patch16_224'
MODEL_PATH=''
OUTPUT_PREFIX=''


# Set the path to save video
# idx: the position index of the patch within a frame.
# idx_f: the index of the frame.
OUTPUT_DIR=$OUTPUT_PREFIX
CUDA_VISIBLE_DEVICES=0 python3 run_vis.py \
   --mask_ratio 0.0 \
   --num_k 10\
   --idx 118\
   --idx_f 0\
   --mask_type tube \
   --decoder_depth 0 \
   --model $AREC \
   ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}