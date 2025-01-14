export CUDA_VISIBLE_DEVICES=0

python train.py \
--task "tsp" \
--project_name "consistency_co_test" \
--wandb_logger_name "tsp_50" \
--do_test \
--storage_path "./" \
--test_split "data/tsp/tsp50_concorde_5.68759.txt" \
--inference_schedule "cosine" \
--inference_diffusion_steps 1 \
--two_opt_iterations 0 \
--ckpt_path 'tsp50.ckpt' \
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
