export CUDA_VISIBLE_DEVICES=0,1,2

python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_50_alpha_0.5_not_same_trial_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results/" \
  --training_split "data/tsp50_train/tsp50_uniform_1.28m.txt" \
  --validation_split "data/tsp/tsp50_concorde.txt" \
  --test_split "data/tsp/tsp50_concorde.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 1280 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --rewrite \
  --ckpt_path "ckpts/tsp50_categorical.ckpt" \
  --resume_weight_only