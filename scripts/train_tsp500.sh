export CUDA_VISIBLE_DEVICES=1,2,3

python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_500_alpha_0.5_bz6_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "data/tsp500_train/tsp500_uniform_train_lkh5w.txt" \
  --validation_split "data/tsp/tsp500_concorde.txt" \
  --test_split "data/tsp/tsp500_concorde.txt" \
  --ckpt_path "tsp100.ckpt" \
  --resume_weight_only \
  --sparse_factor 50 \
  --batch_size 6 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency \
  --rewrite