 python search.py \
                    --batch-size 4  \
                    --dataset sceneflow \
                    --crop_height 192  \
                    --crop_width 384 \
                    --gpu-ids [0,1,2,3] \
                    --fea_num_layers 6 \
                    --mat_num_layers 12 \
                    --fea_filter_multiplier 4 --fea_block_multiplier 3 --fea_step 3  \
                    --mat_filter_multiplier 4 --mat_block_multiplier 3 --mat_step 3  \
                    --alpha_epoch 3  \
                    --lr 1e-3 \
                    --testBatchSize 8
                    #--resume './run/sceneflow/experiment_52/checkpoint.pth.tar' 
                    #--debug


