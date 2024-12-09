export CUDA_VISIBLE_DEVICES=0


step_list="5,5"

for step in $step_list; do
    infer=$(echo $step | cut -d',' -f1)
    re=$(echo $step | cut -d',' -f2)
    python train.py \
      --task "mis" \
      --project_name "consistency_co_test" \
      --wandb_logger_name "mis_er" \
      --do_test \
      --storage_path "./" \
      --test_split "data/mis/er_1400_1600_0.15_test/*gpickle" \
      --inference_schedule "cosine" \
      --inference_diffusion_steps $infer \
      --ckpt_path "cns_ckpts/mis_er.ckpt" \
      --resume_weight_only \
      --parallel_sampling 1 \
      --sequential_sampling 1 \
      --consistency \
      --hidden_dim 128 \
      --rewrite \
      --guided \
      --rewrite_steps $re \
      --rewrite_ratio 0.2 \
      --c1 2 \
      --c2 2 \
      --offline > 'cns_er_1400_1600_g_'$infer'_'$re'.txt'


      python train.py \
      --task "mis" \
      --project_name "consistency_co_test" \
      --wandb_logger_name "mis_er" \
      --do_test \
      --storage_path "./" \
      --test_split "data/mis/er_1400_1600_0.15_test/*gpickle" \
      --inference_schedule "cosine" \
      --inference_diffusion_steps $infer \
      --ckpt_path "cns_ckpts/mis_er.ckpt" \
      --resume_weight_only \
      --parallel_sampling 1 \
      --sequential_sampling 4 \
      --consistency \
      --hidden_dim 128 \
      --rewrite \
      --guided \
      --rewrite_steps $re \
      --rewrite_ratio 0.2 \
      --c1 2 \
      --c2 2 \
      --offline > 'cns_er_1400_1600_s_'$infer'_'$re'.txt'
done