CUDA_VISIBLE_DEVICES=1 python predict.py \
                --sceneflow=1    --maxdisp=192 \
                --crop_height=576  --crop_width=960  \
                --data_path='/store/Datasets/flow/SceneFlow/' \
                --test_list='./dataloaders/lists/sceneflow_test.list' \
                --save_path='./predict/sceneflow/images/' \
                \
                --fea_num_layer 6 \
                --mat_num_layers 12 \
                --fea_filter_multiplier 8 \
                --fea_block_multiplier 4 \
                --fea_step 3  \
                --mat_filter_multiplier 8 \
                --mat_block_multiplier 4 \
                --mat_step 3  \
                \
                --net_arch_fea='run01/sceneflow/experiment_27/feature_network_path.npy' \
                --cell_arch_fea='run01/sceneflow/experiment_27/feature_genotype.npy' \
                --net_arch_mat='run01/sceneflow/experiment_27/matching_network_path.npy' \
                --cell_arch_mat='run01/sceneflow/experiment_27/matching_genotype.npy' \
                \
                --resume='./run01/sceneflow/experiment_27/checkpoint_2.pth.tar' \
                --debug

