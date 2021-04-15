if [ "$1" != "" ]; then
    download_path="$1"
else
    download_path="/home/blarson/Downloads"
fi

if [ "$2" != "" ]; then
    dataset_path="$2"
else
    dataset_path="/store/Datasets/flow/SceneFlow"
fi

# Sceneflow dataset URLs: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/all_download_paths.txt
                                                                                                                                                                                                                         
if [ ! -f "${download_path}/flyingthings3d__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/FlyingThings3D/raw_data/flyingthings3d__camera_data.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/flyingthings3d__camera_data.tar
fi

if [ ! -f "${download_path}/driving__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/Driving/raw_data/driving__camera_data.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/driving__camera_data.tar
fi

if [ ! -f "${download_path}/monkaa__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/Monkaa/raw_data/monkaa__camera_data.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/monkaa__camera_data.tar
fi

if [ ! -f "${download_path}/monkaa__frames_finalpass.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/monkaa__frames_finalpass.tar
fi

if [ ! -f "${download_path}/monkaa__disparity.tar.bz2" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2 -P ${download_path}
    tar -C ${dataset_path} -xvjf ${download_path}/monkaa__disparity.tar.bz2
fi

if ! command -v transmission-cli &> /dev/null
then
    sudo apt install transmission-cli
fi

if [ ! -f "${download_path}/flyingthings3d__disparity.tar.bz2" ]; then
    # Torent file invalid.  Go to source
    # transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2.torrent -w ${download_path}
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2  -P ${download_path}
    tar -C ${dataset_path} -xvjf ${download_path}/flyingthings3d__disparity.tar.bz2
fi

if [ ! -f "${download_path}/flyingthings3d__frames_finalpass.tar" ]; then
    # transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar.torrent -w ${download_path}
    wget http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/flyingthings3d__frames_finalpass.tar
fi

if [ ! -f "${download_path}/driving__frames_finalpass.tar" ]; then
    # transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar.torrent -w ${download_path}
    wget http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/driving__frames_finalpass.tar
fi

if [ ! -f "${download_path}/driving__disparity.tar.bz2" ]; then
    # transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2.torrent -w ${download_path}
    wget http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2 -P ${download_path}
    tar -C ${dataset_path} -xvf ${download_path}/driving__disparity.tar.bz2
fi
