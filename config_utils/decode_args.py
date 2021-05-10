import argparse

def obtain_decode_args():
    parser = argparse.ArgumentParser(description="LEStereo Decoding..")
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'],
                        help='dataset name (default: sceneflow)') 
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    
    # Runtime Parameters
    parser.add_argument('--debug', action='store_true', help='True, enable debug and stop at breakpoint')
    parser.add_argument('--debug_port', type=int, default=3000, help='Debug port')

    return parser.parse_args()
