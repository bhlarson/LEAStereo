import argparse

def obtain_convert_args():

    parser = argparse.ArgumentParser(description='LEStereo Prediction')
    parser.add_argument('--crop_height', type=int, default=576, help="crop height")
    parser.add_argument('--crop_width', type=int, default=960, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='./run/sceneflow/best/checkpoint/best.pth', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
    parser.add_argument('--data_path', type=str, default='/store/Datasets/flow/SceneFlow/', help="data root")
    parser.add_argument('--test_list', type=str, default='./dataloaders/lists/sceneflow_test.list', help="training list")
    parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
    ######### LEStereo params####################
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--mat_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default='run/sceneflow/best/architecture/feature_network_path.npy', type=str)
    parser.add_argument('--cell_arch_fea', default='run/sceneflow/best/architecture/feature_genotype.npy', type=str)
    parser.add_argument('--net_arch_mat', default='run/sceneflow/best/architecture/matching_network_path.npy', type=str)
    parser.add_argument('--cell_arch_mat', default='run/sceneflow/best/architecture/matching_genotype.npy', type=str)

    # Runtime Parameters
    parser.add_argument('--debug', action='store_true', help='True, enable debug and stop at breakpoint')
    parser.add_argument('--debug_port', type=int, default=3000, help='Debug port')

    args = parser.parse_args()
    return args