export CUDA_VISIBLE_DEVICES=1

step_list="5 5"
for step in $step_list; do
    infer=$(echo $step | cut -d',' -f1)
    re=$(echo $step | cut -d',' -f2)
    python train.py \
      --task "mis" \
      --project_name "consistency_co_test" \
      --wandb_logger_name "mis_sat" \
      --do_test \
      --storage_path "./" \
      --test_split "data/mis/sat/test/*gpickle" \
      --inference_schedule "cosine" \
      --inference_diffusion_steps $infer \
      --test_split_label_dir "data/mis/sat/test_labels" \
      --ckpt_path "/mnt/nas2/home/guojinpei/consistency-co/results/mis_consistency_sat_pretrain/0nirz9t4/checkpoints/last.ckpt" \
      --resume_weight_only \
      --parallel_sampling 1 \
      --sequential_sampling 1 \
      --consistency \
      --rewrite \
      --guided \
      --rewrite_steps $re \
      --rewrite_ratio 0.5 \
      --c1 1 \
      --c2 20 \
      --offline > 'cns_sat_g_'$infer'_'$re'.txt'

      python train.py \
      --task "mis" \
      --project_name "consistency_co_test" \
      --wandb_logger_name "mis_sat" \
      --do_test \
      --storage_path "./" \
      --test_split "data/mis/sat/test/*gpickle" \
      --inference_schedule "cosine" \
      --inference_diffusion_steps $infer \
      --test_split_label_dir "data/mis/sat/test_labels" \
      --ckpt_path "/mnt/nas2/home/guojinpei/consistency-co/results/mis_consistency_sat_pretrain/0nirz9t4/checkpoints/last.ckpt" \
      --resume_weight_only \
      --parallel_sampling 4 \
      --sequential_sampling 1 \
      --consistency \
      --rewrite \
      --guided \
      --rewrite_steps $re \
      --rewrite_ratio 0.5 \
      --c1 1 \
      --c2 20 \
      --offline > 'cns_sat_s_'$infer'_'$re'.txt'
done
