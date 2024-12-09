export CUDA_VISIBLE_DEVICES=3

step_list="1,0 5,0"
for step in $step_list; do
    infer=$(echo $step | cut -d',' -f1)
    re=$(echo $step | cut -d',' -f2)

    python train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_10000" \
       --do_test \
       --storage_path "./" \
       --test_split "data/tsp/tsp10000_concorde.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 0 \
       --ckpt_path 'cns_ckpts/tsp10000.ckpt' \
       --consistency \
       --sparse_factor 100 \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > cns_10000_g_$infer'_'$re'.txt'

    python train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_10000" \
       --do_test \
       --storage_path "./" \
       --test_split "data/tsp/tsp10000_concorde.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 5000 \
       --ckpt_path 'cns_ckpts/tsp10000.ckpt' \
       --consistency \
       --sparse_factor 100 \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > cns_two_opt_10000_g_$infer'_'$re'.txt'

#    python train.py \
#       --task "tsp" \
#       --project_name "consistency_co_test" \
#       --wandb_logger_name "tsp_1000" \
#       --do_test \
#       --storage_path "./" \
#       --test_split "data/tsp_test/tsp1000_concorde_23.11812.txt" \
#       --inference_schedule "cosine" \
#       --inference_diffusion_steps $infer \
#       --two_opt_iterations 0 \
#       --ckpt_path 'cns_ckpts/tsp1000.ckpt' \
#       --consistency \
#       --sparse_factor 100 \
#       --resume_weight_only \
#       --parallel_sampling 2 \
#       --sequential_sampling 2 \
#       --rewrite \
#       --guided \
#       --rewrite_steps $re \
#       --rewrite_ratio 0.2 \
#       --offline > cns_1000_s_$infer'_'$re'.txt'
#
#    python train.py \
#       --task "tsp" \
#       --project_name "consistency_co_test" \
#       --wandb_logger_name "tsp_1000" \
#       --do_test \
#       --storage_path "./" \
#       --test_split "data/tsp_test/tsp1000_concorde_23.11812.txt" \
#       --inference_schedule "cosine" \
#       --inference_diffusion_steps $infer \
#       --two_opt_iterations 5000 \
#       --ckpt_path 'cns_ckpts/tsp1000.ckpt' \
#       --consistency \
#       --sparse_factor 100 \
#       --resume_weight_only \
#       --parallel_sampling 2 \
#       --sequential_sampling 2 \
#       --rewrite \
#       --guided \
#       --rewrite_steps $re \
#       --rewrite_ratio 0.2 \
#       --offline > cns_two_opt_1000_s_$infer'_'$re'.txt'
done