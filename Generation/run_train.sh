# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

# DATASET=Yelp
# OUTPUT_DIR=./ckpt/$DATASET/

DATASET=Arts
OUTPUT_DIR=./ckpt/$DATASET/
torchrun --nproc_per_node=2 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --base_model /home/liuwei/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
    --epochs 200 \
    --index_file Arts.index.epoch10000.alpha1e-1-beta1e-4-20251105_163545.json \
    --user_index_file Arts.index.epoch10000.alpha1e-1-beta1e-4-20251105_163946.json\
    --temperature 1.0

# DATASET=Beauty
# OUTPUT_DIR=./ckpt/$DATASET/

# torchrun --nproc_per_node=2  --master_port=2315 ./finetune.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --per_device_batch_size 256 \
#     --learning_rate 2e-4 \
#     --base_model /home/liuwei/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
#     --epochs 200 \
#     --index_file Beauty.index.epoch10000.alpha1e-1-beta1e-4-20250927_124432.json \
#     --user_index_file Beauty.index.user_20250927_233633.json \
#     --temperature 0.7

# DATASET=Instruments
# OUTPUT_DIR=./ckpt/$DATASET/
# export CUDA_VISIBLE_DEVICES=0,1   
# python ./finetune.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --base_model /home/liuwei/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
#     --per_device_batch_size 256 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file Instruments.index.epoch10000.alpha1e-1-beta1e-4-20250905_093648.json \
#     --user_index_file Instruments.index.user_mean_llama-20251021_120444.json \
#     --temperature 1.0
 