export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  --task "mis" \
  --project_name "consistency_co" \
  --wandb_logger_name "mis_consistency_sat_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "data/mis/sat/train/*gpickle" \
  --training_split_label_dir "data/mis/sat/train_labels" \
  --validation_split "data/mis/sat/test/*gpickle" \
  --validation_split_label_dir "data/mis/sat/test_labels" \
  --ckpt_path "ckpts/mis_sat_categorical.ckpt" \
  --resume_weight_only \
  --use_activation_checkpoint \
  --batch_size 16 \
  --num_epochs 100 \
  --validation_examples 500 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency
