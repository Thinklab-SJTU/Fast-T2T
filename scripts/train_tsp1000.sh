export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_1000_alpha_0.5_bz4_occupy" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "/mnt/nas/dataset_share/tsp1000_uniform/tsp1000_uniform_lkh10w.txt" \
  --validation_split "tsp1000_concorde.txt" \
  --use_activation_checkpoint \
  --sparse_factor 100 \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency \
  --rewrite