python ./CSGT/main.py \
  --device cuda:0 \
  --data_path ./CSGT/data/Arts/Arts_user_time_decay_representation.npy \
  --cf_emb  ./CSGT/data/Arts/Arts_user_cf_representations.npy \
  --ckpt_dir ./checkpoint/\
  --alpha 0.01 \
  --use_constrain_loss true\
  --beta 0.0001 \
  --use_knn true\
  --knn_weight 0.01