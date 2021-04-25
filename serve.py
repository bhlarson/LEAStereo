#%%
#!/usr/bin/env python
import os, sys, json, argparse, math
import configparser
import wget
import cv2
import numpy as np
from flask import Flask, render_template, Response
from datetime import datetime
import shutil
import torch
from time import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from retrain.LEAStereo import LEAStereo 
import pyzed.sl as sl
from base_camera import BaseCamera

#sys.path.insert(0, os.path.abspath(''))

config = {
      'area_filter_min': 250,
      'size_divisible': 32,
      }

app = Flask(__name__)

model = None 
infer = None
args = None

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',help='Wait for debugge attach')
    parser.add_argument('--debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('--zed_sn', type=int, default=26641093, help='ZED camera serial number for camera alignment')
    parser.add_argument('--crop_height', type=int, default=576, help="crop height")
    parser.add_argument('--crop_width', type=int, default=960, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--resume', type=str, default='./run/sceneflow/best/checkpoint/best.pth', help="resume from saved model")
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

    loaded_args = parser.parse_args()
    return loaded_args

def download_calibration_file(serial_number) :
    if os.name == 'nt' :
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else :
        hidden_path = './calibration/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url+str(serial_number), out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file

def init_calibration(calibration_file, image_size) :

    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])

    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)

    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Camera(BaseCamera):
    video_source = 0
    config = None
    model = None
    cap = None

    def __init__(self, config):
        Camera.config = config

        torch.backends.cudnn.benchmark = True
        Camera.image_size = Resolution()
        calibration_file = download_calibration_file(args.zed_sn)
        camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, Camera.image_size)
        Camera.camera_matrix_left = camera_matrix_left
        Camera.camera_matrix_right = camera_matrix_right
        Camera.map_left_x = map_left_x
        Camera.map_left_y = map_left_y
        Camera.map_right_x = map_right_x
        Camera.map_right_y = map_right_y

        # Open the ZED camera
        Camera.cap = cv2.VideoCapture(0)
        if Camera.cap.isOpened() == 0:
            exit(-1)

        # Set the video resolution to HD720
        Camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Camera.image_size.width*2)
        Camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Camera.image_size.height)

        if args.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        print('===> Building LEAStereo model')
        Camera.model = LEAStereo(args)

        if args.cuda:
            Camera.model = torch.nn.DataParallel(Camera.model).cuda()

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                Camera.model.load_state_dict(checkpoint['state_dict'], strict=True)      
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


            super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        while True:
            frame = np.zeros([10, 10, 3], dtype=np.uint8)
            yield frame

def crop(image, out_size):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = in_size[0] - out_size[0]
    w_diff = in_size[1] - out_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be longer than or equal to the output size'

    if h_diff > 0 and w_diff > 0:
        h_idx = h_diff//2
        w_idx = w_diff//2
        image = image[h_idx:h_idx + out_size[0], w_idx:w_idx + out_size[1]]
    elif h_diff > 0:
        h_idx = h_diff//2
        image = image[h_idx:h_idx + out_size[0], :]
    elif w_diff > 0:
        w_idx = w_diff//2
        image = image[:, w_idx:w_idx + out_size[1]]

    return image

def zero_pad(image, out_size):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = out_size[0] - in_size[0]
    w_diff = out_size[1] - in_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be shorter than or equal to the output size'

    out_size_max = [max(out_size[0], in_size[0]), max(out_size[1], in_size[1])]
    if len(image.shape) > 2:
        out_size_max.append(image.shape[2])
    image_out = np.zeros(out_size_max, dtype=image.dtype)

    if h_diff > 0 and w_diff > 0:
        h_idx = h_diff//2
        w_idx = w_diff//2
        image_out[h_idx:h_idx + in_size[0], w_idx:w_idx + in_size[1]] = image
    elif h_diff > 0:
        h_idx = h_diff//2
        image_out[h_idx:h_idx + in_size[0], :] = image
    elif w_diff > 0:
        w_idx = w_diff//2
        image_out[:, w_idx:w_idx + in_size[1]] = image
    else:
        image_out = image

    return image_out

def resize_with_crop_or_pad(image, out_size):
    if image.shape[0] > out_size[0] or image.shape[1] > out_size[1]:
        image = crop(image, out_size)
    if image.shape[0] < out_size[0] or image.shape[1] < out_size[1]:
        image = zero_pad(image, out_size)

    return image

def img_normalize(img):
    size = np.shape(img)
    height = size[0]
    width = size[1]

    img_cwh = np.zeros([3, height, width], 'float32')
    img = np.asarray(img)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    #Normalize each color and reorde pixels from WHC to CHW
    img_cwh[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    img_cwh[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    img_cwh[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])

    img = np.ones([1, 3,height,width],'float32')
    img[0, :, :, :] = img_cwh[0: 3, :, :]

    return torch.from_numpy(img).float(), height, width

class Resolution :
    width = 1280
    height = 720


@app.route('/') 
def index():
    """Video streaming home page."""
    return render_template('index.html')

def CropOrigonal(image, height, width):
    return image[:height,:width,:]
    #return tf.image.crop_to_bounding_box(image, 0, 0, height, width

def gen(camera):
    """Video streaming generator function."""

    while True:

        tbefore = datetime.now()


        # Retrieve image
        retval, frame = Camera.cap.read()

        left_right_image = np.split(frame, 2, axis=1)

        # Apply camera calibrations
        imageL = cv2.remap(left_right_image[0], Camera.map_left_x, Camera.map_left_y, interpolation=cv2.INTER_LINEAR)
        imageR = cv2.remap(left_right_image[1], Camera.map_right_x, Camera.map_right_y, interpolation=cv2.INTER_LINEAR)

        imageL = resize_with_crop_or_pad(imageL, [args.crop_height, args.crop_width])
        imageR = resize_with_crop_or_pad(imageR, [args.crop_height, args.crop_width])

        imageL, _, _ = img_normalize(imageL)
        imageR, _, _ = img_normalize(imageR)

        imageL = Variable(imageL, requires_grad = False)
        imageR = Variable(imageR, requires_grad = False)

        camera.model.eval()
        if args.cuda:
            imageL = imageL.cuda()
            imageR = imageR.cuda()
        torch.cuda.synchronize()
        start_time = time()
        with torch.no_grad():
            prediction = camera.model(imageL, imageR)
        torch.cuda.synchronize()
        prediction = prediction.cpu()
        depth = prediction.detach().numpy()
        depth = np.squeeze(depth).astype(np.uint8)
        tPredict = datetime.now()

        tAfter = datetime.now()
        dInfer = tPredict-tbefore
        dImAn = tAfter-tPredict

        #outputs['pred_age'].numpy()
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
        font = cv2.FONT_HERSHEY_SIMPLEX
        resultsDisplay = 'infer:{:.3f}s display:{:.3f}s'.format(dInfer.total_seconds(), dImAn.total_seconds())
        cv2.putText(depth, resultsDisplay, (10,25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # encode as a jpeg image and return it
        frame = cv2.imencode('.jpg', depth)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera(config)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    args = load_args()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
        # Allow other computers to attach to ptvsd at this IP address and port.
        debugpy.listen(address=('0.0.0.0', args.debug_port))

        # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()
        print("Debugger attached")


    app.run(host='0.0.0.0', threaded=True)
