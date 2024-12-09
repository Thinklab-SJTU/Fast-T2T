#python rb_graph/rbgraph_generator.py --num_graph 500 --save_dir /mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test && python -u main.py solve \
#    kamis \
#    /mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test\
#    /mnt/nas2/home/guojinpei/consistency-co/data/mis/rb/rb200_300_test_label \

#i=2
#python rb_graph/rbgraph_generator.py --num_graph 18000 --save_dir /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train_$i && python -u main.py solve \
#    kamis \
#    /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train_$i \
#    /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train_label_$i

python rb_graph/rbgraph_generator.py --num_graph 90000 --save_dir /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train && python -u main.py solve \
    kamis \
    /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train \
    /mnt/nas/dataset_share/guojinpei/mis/rb/rb200_300_train_label