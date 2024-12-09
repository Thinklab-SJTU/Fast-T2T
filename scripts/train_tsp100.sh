export CUDA_VISIBLE_DEVICES=0,1,2

python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_100_alpha_0.5_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results/" \
  --training_split "/mnt/nas/dataset_share/tsp100_uniform/tsp100_uniform_1.28m.txt" \
  --validation_split "data/tsp/tsp100_concorde.txt" \
  --ckpt_path "/home/guojinpei/consistency-co/results/tsp_consistency_100_alpha_0.5_pretrain/qiw53di0/checkpoints/last.ckpt" \
  --batch_size 12 \
  --num_epochs 50 \
  --validation_examples 1280 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --rewrite
