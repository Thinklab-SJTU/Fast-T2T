cd data/mis-benchmark-framework || exit

#python -u main.py gendata \
#    random \
#    None \
#    /data/guojinpei/consistency-co/data/mis/er/er_700_800_train \
#    --model er \
#    --min_n 700 \
#    --max_n 800 \
#    --num_graphs 163840 \
#    --er_p 0.15

mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

python -u main.py \
    solve \
    kamis \
    /data/guojinpei/consistency-co/debug/ \
    /data/guojinpei/consistency-co/debug_labels \
    --time_limit 60