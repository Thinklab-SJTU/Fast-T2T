export CUDA_VISIBLE_DEVICES=0

python train.py \
--task "tsp" \
--project_name "consistency_co_test" \
--wandb_logger_name "tsp_100" \
--do_test \
--storage_path "./" \
--test_split "data/tsp/tsp100_concorde_7.75585.txt" \
--inference_schedule "cosine" \
--inference_diffusion_steps 1 \
--two_opt_iterations 0 \
--ckpt_path 'tsp100.ckpt' \
--consistency \
--use_intermediate \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--rewrite \
--guided \
--rewrite_steps 1 \
--rewrite_ratio 0.2 \
--offline
