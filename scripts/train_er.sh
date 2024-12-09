export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

python -u train.py \
  --task "mis" \
  --project_name "consistency_co" \
  --wandb_logger_name "mis_consistency_er_128" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "/mnt/nas/dataset_share/guojinpei/mis/er/er_700_800_train/*gpickle" \
  --training_split_label_dir "/mnt/nas/dataset_share/guojinpei/mis/er/er_700_800_train_labels" \
  --validation_split "/mnt/nas/dataset_share/guojinpei/mis/er/er_700_800_test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --hidden_dim 128 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --use_activation_checkpoint \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency