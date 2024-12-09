export CUDA_VISIBLE_DEVICES=2,7

python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_10000" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --ckpt_path "/root/consistency-co/results/tsp_consistency_10000/fmqv8joa/checkpoints/last.ckpt" \
  --training_split "tsp10000_uniform_alkh_6.4k_train.txt" \
  --validation_split "data/tsp/tsp10000_concorde.txt" \
  --test_split "data/tsp/tsp10000_concorde.txt" \
  --use_activation_checkpoint \
  --sparse_factor 100 \
  --batch_size 2 \
  --fp16 \
  --num_epochs 50 \
  --validation_examples 16 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency \
  --rewrite \
