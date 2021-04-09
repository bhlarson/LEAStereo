if [ "$1" != "" ]; then
    download_path="$1"
else
    download_path="~/Downloads"
fi

if [ "$2" != "" ]; then
    dataset_path="$2"
else
    dataset_path="/store/Datasets/flow/SceneFlow"
fi
                                                                                                                                                                                                                         
if [ ! -f "flyingthings3d__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/FlyingThings3D/raw_data/flyingthings3d__camera_data.tar -P ${download_path}
fi

tar -C ${dataset_path} -xvf flyingthings3d__camera_data.tar

if [ ! -f "driving__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/Driving/raw_data/driving__camera_data.tar -P ${download_path}
fi

tar -C ${dataset_path} -xvf driving__camera_data.tar

if [ ! -f "monkaa__camera_data.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/Monkaa/raw_data/monkaa__camera_data.tar -P ${download_path}
fi
tar -C ${dataset_path} -xvf monkaa__camera_data.tar

if [ ! -f "monkaa__frames_finalpass.tar" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar -P ${download_path}
fi
tar -C ${dataset_path} -xvf monkaa__frames_finalpass.tar

if [ ! -f "monkaa__disparity.tar.bz2" ]; then
    wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2 -P ${download_path}
fi
tar -C ${dataset_path} -xvjf monkaa__disparity.tar.bz2

if ! command -v transmission &> /dev/null
then
    sudo apt install transmission-cli
fi

if [ ! -f "flyingthings3d__frames_finalpass.tar" ]; then
    transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar.torrent -w ${download_path}
fi
tar -C ${dataset_path} -xvf flyingthings3d__frames_finalpass.tar

if [ ! -f "flyingthings3d__disparity.tar.bz2" ]; then
    transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2.torrent -w ${download_path}
fi
tar -C ${dataset_path} -xvjf flyingthings3d__disparity.tar.bz2

if [ ! -f "driving__frames_finalpass.tar" ]; then
    transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar.torrent -w ${download_path}
fi
tar -C ${dataset_path} -xvf driving__frames_finalpass.tar

if [ ! -f "driving__disparity.tar.bz2" ]; then
    transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2.torrent -w ${download_path}
fi
tar -C ${dataset_path} -xvf driving__disparity.tar.bz2

if [ ! -f "flyingthings3d__disparity.tar.bz2" ]; then
    transmission-cli https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2.torrent -w ${download_path}
fi
tar -C ${dataset_path} -xjf flyingthings3d__disparity.tar.bz2