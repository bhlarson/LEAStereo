from __future__ import print_function
import os
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from retrain.LEAStereo import LEAStereo 

from config_utils.convert_args import obtain_convert_args
from utils.colorize import get_color_map
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from time import time
from struct import unpack
import matplotlib.pyplot as plt
import re
import numpy as np
import pdb
from path import Path

opt = obtain_convert_args()
print(opt)

if opt.debug:
    print("Wait for debugger attach")
    import debugpy
    # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
    # Launch applicaiton on remote computer: 
    # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
    # Allow other computers to attach to ptvsd at this IP address and port.
    debugpy.listen(address=('0.0.0.0', opt.debug_port))
    # Pause the program until a remote debugger is attached

    debugpy.wait_for_client()
    print("Debugger attached")

torch.backends.cudnn.benchmark = True

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt)

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))

mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = model.cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)      
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

dummy_input = torch.randn(1, 3, opt.crop_height, opt.crop_width, device='cuda')
torch.onnx.export(model, dummy_input, "LEAStereo.onnx", verbose=True, input_names=['left_image', 'right_image'], output_names=['distance_image'])

'''  
turbo_colormap_data = get_color_map()


def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata=[]; gdata=[]; bdata=[]
    for istep in range(nsteps):
        r = rgbdata[istep,0]
        g = rgbdata[istep,1]
        b = rgbdata[istep,2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red':   rdata,
                 'green': gdata,
                 'blue':  bdata}

    return mpl_data

mpl_data = RGBToPyCmap(turbo_colormap_data)
plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])

def readPFM(file): 
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)

    return img, height, width

def save_pfm(filename, image, scale=1):
    
    # Save a Numpy array to a PFM file.
    
    color = None
    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


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


def load_data(leftname, rightname, crop_height, crop_width):
    left = np.array(Image.open(leftname).convert('RGB'))
    left = resize_with_crop_or_pad(left, [crop_height, crop_width])

    right = np.array(Image.open(rightname).convert('RGB'))
    right = resize_with_crop_or_pad(right, [crop_height, crop_width])

    return left, right

def test_md(leftname, rightname, savename, imgname):
    left, right = load_data(leftname, rightname, opt.crop_height, opt.crop_width)
    input1, height, width = img_normalize(left)
    input2, _, _ = img_normalize(right)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)
    torch.cuda.synchronize()
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    imDisparity = prediction.cpu()
    imDisparity = imDisparity.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        imDisparity = imDisparity[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        imDisparity = imDisparity[0, :, :]
    plot_disparity(imgname, imDisparity, 192)
    savepfm_path = savename.replace('.png','') 
    imDisparity = np.flipud(imDisparity)

    disppath = Path(savepfm_path)
    disppath.makedirs_p()
    save_pfm(savepfm_path+'/disp0LEAStereo.pfm', imDisparity, scale=1)
    ##########write time txt########
    fp = open(savepfm_path+'/timeLEAStereo.txt', 'w')
    runtime = "XXs"  
    fp.write(runtime)   
    fp.close()

def test_kitti(leftname, rightname, savename):
    left, right = load_data(leftname, rightname, opt.crop_height, opt.crop_width)
    input1, height, width = img_normalize(left)
    input2, _, _ = img_normalize(right)
 
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():        
        prediction = model(input1, input2)
        
    imDisparity = prediction.cpu()
    imDisparity = imDisparity.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        imDisparity = imDisparity[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        imDisparity = imDisparity[0, :, :]
    skimage.io.imsave(savename, (imDisparity * 256).astype('uint16'))


def test(leftname, rightname, savename): 
    left, right = load_data(leftname, rightname, opt.crop_height, opt.crop_width)
    input1, height, width = img_normalize(left)
    input2, _, _ = img_normalize(right)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    imDisparity = prediction.cpu()
    imDisparity = imDisparity.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        imDisparity = imDisparity[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        imDisparity = imDisparity[0, :, :]
    plot_disparity(savename, imDisparity, 192, left, right)
    savename_pfm = savename.replace('png','pfm') 
    imDisparity = np.flipud(imDisparity)

def plot_disparity(savename, imDisparity, max_disp, left, right):
    fig = plt.figure(figsize=(8.5, 11))  # width, height in inches
    sub = fig.add_subplot(3, 1, 1)
    sub.title.set_text('Left Image')
    sub.imshow(left, interpolation='nearest')
    plt.axis('off')
    sub = fig.add_subplot(3, 1, 2)
    sub.title.set_text('Right Image')
    sub.imshow(right, interpolation='nearest')
    plt.axis('off')
    sub = fig.add_subplot(3, 1, 3)
    sub.title.set_text('Distance Image')
    sub.imshow(imDisparity, interpolation='nearest', vmin=0, vmax=max_disp, cmap='turbo')
    plt.axis('off')
    fig.savefig(savename, bbox_inches='tight')

    #plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')

   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    for index in range(len(filelist)):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            test_kitti(leftname, rightname, savename)

        if opt.kitti2012:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            test_kitti(leftname, rightname, savename)

        if opt.sceneflow:
            leftname = file_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'frames_finalpass/' + current_file[0: len(current_file) - 14] + 'right/' + current_file[len(current_file) - 9:len(current_file) - 1]
            leftgtname = file_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            disp_left_gt, height, width = readPFM(leftgtname)
            savenamegt = opt.save_path + "{:d}_gt.png".format(index)

            imleft = np.array(Image.open(leftname).convert('RGB'))
            imleft = resize_with_crop_or_pad(imleft, [opt.crop_height, opt.crop_width])
            imright = np.array(Image.open(rightname).convert('RGB'))
            imright = resize_with_crop_or_pad(imright, [opt.crop_height, opt.crop_width])
            disp_left_gt = resize_with_crop_or_pad(disp_left_gt, [opt.crop_height, opt.crop_width])
            plot_disparity(savenamegt, disp_left_gt, 192, imleft, imright)

            savename = opt.save_path + "{:d}.png".format(index)
            test(leftname, rightname, savename)

        if opt.middlebury:
            leftname = file_path + current_file[0: len(current_file) - 1]
            rightname = leftname.replace('im0','im1') 

            temppath = opt.save_path.replace(opt.save_path.split("/")[-2], opt.save_path.split("/")[-2]+"/images")     
            img_path = Path(temppath)
            img_path.makedirs_p()
            savename = opt.save_path + current_file[0: len(current_file) - 9] + ".png"
            img_name = img_path + current_file[0: len(current_file) - 9] + ".png"
            test_md(leftname, rightname, savename, img_name)
'''
