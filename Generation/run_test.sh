DATASET=Arts
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx_$(date +"%Y%m%d_%H%M%S").json
CKPT_PATH=./ckpt/Arts_1/checkpoint-6750

python test.py \
    --gpu_id 1 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 128 \
    --num_beams 20 \
    --base_model /home/liuwei/.cache/huggingface/hub/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
    --test_prompt_ids 0 \
    --index_file Arts.index.epoch10000.alpha1e-1-beta1e-4-20251105_163545.json \
    --user_index_file Arts.index.epoch10000.alpha1e-1-beta1e-4-20251105_163946.json