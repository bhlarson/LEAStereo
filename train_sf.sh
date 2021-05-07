python3 train.py --batch_size=4 \
                --crop_height=384 \
                --crop_width=576 \
                --maxdisp=192 \
                --threads=8 \
                --save_path='./run/sceneflow/experiment_3/checkpoint_0/retrain/' \
                --fea_num_layer 6 --mat_num_layers 12 \
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='./run/sceneflow/experiment_3/checkpoint_0/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/experiment_3/checkpoint_0/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/experiment_3/checkpoint_0/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/experiment_3/checkpoint_0/matching_genotype.npy' \
                --nEpochs=20 2>&1 |tee ./run/sceneflow/experiment_3/checkpoint_0/retrain/log.txt

