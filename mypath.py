class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sceneflow':
            return '/store/Datasets/flow/SceneFlow/'
        elif dataset == 'kitti15':
            return 'store/Datasets/flow/kitti2015/training/'
        elif dataset == 'kitti12':
            return 'store/Datasets/flow/kitti2012/training/'
        elif dataset == 'middlebury':
            return 'store/Datasets/flow/MiddEval3/trainingH/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
