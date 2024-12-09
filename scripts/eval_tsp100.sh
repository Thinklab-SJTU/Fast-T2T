export CUDA_VISIBLE_DEVICES=2


for infer in 1 2 3
do
  for re in 1 2 3
    do
      python train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_100" \
       --do_test \
       --storage_path "./" \
       --test_split "/mnt/nas/dataset_share/majiale/ML4TSP/data/tsp_uniform/tsp100_concorde_7.75585.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 0 \
       --ckpt_path 'tsp100.ckpt' \
       --consistency \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > 100_$infer'_'$re'.txt'

      python train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_100" \
       --do_test \
       --storage_path "./" \
       --test_split "/mnt/nas/dataset_share/majiale/ML4TSP/data/tsp_uniform/tsp100_concorde_7.75585.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 1000 \
       --ckpt_path 'tsp100.ckpt' \
       --consistency \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > two_opt_100_$infer'_'$re'.txt'
    done
done
