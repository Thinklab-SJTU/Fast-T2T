export CUDA_VISIBLE_DEVICES=2

step_list="1,0 5,0 1,1 5,5"
for step in $step_list; do
    infer=$(echo $step | cut -d',' -f1)
    re=$(echo $step | cut -d',' -f2)
    python train.py \
        --task "mis" \
        --project_name "consistency_co_test" \
        --wandb_logger_name "mis_rb" \
        --do_test \
        --storage_path "./" \
        --test_split "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test/*gpickle" \
        --test_split_label_dir "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test_label" \
        --inference_schedule "cosine" \
        --inference_diffusion_steps $infer \
        --ckpt_path "/mnt/nas2/home/guojinpei/consistency-co/results/mis_consistency_rb/zqk7158r/checkpoints/epoch=25-step=292500.ckpt" \
        --resume_weight_only \
        --parallel_sampling 1 \
        --sequential_sampling 1 \
        --rewrite \
        --guided \
        --rewrite_steps $re \
        --rewrite_ratio 0.3 \
        --c1 2 \
        --c2 2 \
        --offline \
        --consistency  > 'cns_rb_g_'$infer'_'$re'.txt'


    python train.py \
      --task "mis" \
      --project_name "consistency_co_test" \
      --wandb_logger_name "mis_rb" \
      --do_test \
      --storage_path "./" \
      --test_split "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test/*gpickle" \
      --test_split_label_dir "/mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test_label" \
      --inference_schedule "cosine" \
      --inference_diffusion_steps $infer \
      --ckpt_path "/mnt/nas2/home/guojinpei/consistency-co/results/mis_consistency_rb/zqk7158r/checkpoints/epoch=25-step=292500.ckpt" \
      --resume_weight_only \
      --parallel_sampling 4 \
      --sequential_sampling 1 \
      --consistency \
      --rewrite \
      --guided \
      --rewrite_steps $re \
      --rewrite_ratio 0.3 \
      --c1 2 \
      --c2 2 \
      --offline > 'cns_rb_s_'$infer'_'$re'.txt'
done
