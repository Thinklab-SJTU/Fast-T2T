cd data/mis-benchmark-framework || exit

mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

python -u main.py \
    solve \
    kamis \
    /home/guojinpei/consistency-co/debug \
    /home/guojinpei/consistency-co/debug_labels \
    --time_limit 120