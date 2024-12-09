export CUDA_VISIBLE_DEVICES=4


for infer in 1 2 3
do
  for re in 1 2 3
    do
#      python train.py \
#       --task "tsp" \
#       --project_name "consistency_co_test" \
#       --wandb_logger_name "tsp_50" \
#       --do_test \
#       --storage_path "./" \
#       --test_split "/mnt/nas/dataset_share/Bench4CO-v2/data/test/tsp50_lkh_500_5.68759.txt" \
#       --inference_schedule "cosine" \
#       --inference_diffusion_steps $infer \
#       --two_opt_iterations 0 \
#       --ckpt_path 'tsp50.ckpt' \
#       --consistency \
#       --resume_weight_only \
#       --parallel_sampling 1 \
#       --sequential_sampling 1 \
#       --rewrite \
#       --guided \
#       --rewrite_steps $re \
#       --rewrite_ratio 0.2 \
#       --offline > 50_$infer'_'$re'.txt'

      python train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_50" \
       --do_test \
       --storage_path "./" \
       --test_split "/mnt/nas/dataset_share/Bench4CO-v2/data/test/tsp50_lkh_500_5.68759.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 5000 \
       --ckpt_path 'tsp50.ckpt' \
       --consistency \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > two_opt_50_$infer'_'$re'.txt'
    done
done