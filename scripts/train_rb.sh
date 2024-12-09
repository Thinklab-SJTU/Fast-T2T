export CUDA_VISIBLE_DEVICES=0,1
#export WANDB_MODE=offline

python train.py \
  --task "mis" \
  --project_name "consistency_co" \
  --wandb_logger_name "mis_consistency_rb" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "/mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train/*gpickle" \
  --training_split_label_dir "/mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train_label" \
  --validation_split "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test/*gpickle" \
  --validation_split_label_dir "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test_label" \
  --validation_examples 128 \
  --batch_size 4 \
  --num_epochs 50 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --num_workers 64 \
  --boundary_func truncate\
  --alpha 0.3 \
  --consistency \

