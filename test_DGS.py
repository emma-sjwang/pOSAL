'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

import numpy as np
import scipy.io as sio
import scipy.misc
from keras.preprocessing import image
from skimage import transform
from skimage.measure import label, regionprops
from time import time
from Utils.utils_get_disc_area import pro_process, BW_img, disc_crop
from PIL import Image
from matplotlib.pyplot import imshow, imsave
from skimage import morphology
from tqdm import tqdm
import cv2
import os
from Model.models import Model_CupSeg, Model_DiscSeg
from keras.applications import imagenet_utils
from skimage.measure import label, regionprops
from Utils.utils import save_img, save_per_img
from Utils.evaluate_segmentation import evaluate_segmentation_results
import random
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)


def calculate_dice(mask_path, prediction_path):
    disc_dice = []
    cup_dice = []
    file_test_list = [file for file in os.listdir(mask_path) if file.lower().endswith('png')]
    for filename in file_test_list:
        mask = np.asarray(image.load_img(os.path.join(mask_path, filename), grayscale=True))
        prediction = np.asarray(image.load_img(os.path.join(prediction_path, filename.split('.')[0]+'.bmp'), grayscale=True))
        mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
        prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],2))
        mask_binary[mask < 200] = [1, 0]
        mask_binary[mask < 50] = [1, 1]
        prediction_binary[prediction < 200] = [1, 0]
        prediction_binary[prediction < 50] = [1, 1]
        disc_dice.append(dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0]))
        cup_dice.append(dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]))
    return sum(disc_dice) / (1. * len(disc_dice)), sum(cup_dice) / (1. * len(cup_dice))


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    ''' parameters '''
    DiscROI_size = 640
    DiscSeg_size = 640
    CDRSeg_size = 512

    data_type = '.png'
    save_data_type = 'png'
    dataset = "Drishti-GS"

    scale = False
    both = True

    phase = "test"
    if dataset == "refuge":
        data_type = '.jpg'
        phase = "test0"

    data_img_path = '../data/' + dataset + '/'+ phase +'/image/'
    data_save_path = './data/' + dataset + '/' + phase + '/OSAL_epoch100/'
    mask_path = '../data/' + dataset + '/'+ phase +'/mask/'

    # change location to wherever you like
    load_from = "./weights/" + dataset + "/DA_patch/Generator/generator_100.h5"

    if not os.path.exists(data_save_path):
        print("Creating save path {}\n\n".format(data_save_path))
        os.makedirs(data_save_path)

    file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
    random.shuffle(file_test_list)
    print(str(len(file_test_list)))

    ''' create model and load weights'''
    DiscSeg_model = Model_DiscSeg(inputsize=DiscSeg_size)
    DiscSeg_model.load_weights('Model_DiscSeg_ORIGA_pretrain.h5')  # download from M-Net

    CDRSeg_model = Model_CupSeg(input_shape = (CDRSeg_size, CDRSeg_size, 3), classes=2, backbone='mobilenetv2')
    CDRSeg_model.load_weights(load_from)

    ''' predict each image '''
    for lineIdx in tqdm(range(0, len(file_test_list))):
        temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
        # load image
        org_img = np.asarray(image.load_img(data_img_path + temp_txt[0]))

        # Disc region detection by U-Net
        temp_img = transform.resize(org_img, (DiscSeg_size, DiscSeg_size, 3))*255
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        [prob_6, prob_7, prob_8, prob_9, prob_10] = DiscSeg_model.predict([temp_img])

        disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
        regions = regionprops(label(disc_map))
        C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
        C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)

        ''' get disc region'''
        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        disc_region_img = disc_region.astype(np.uint8)
        disc_region_img = Image.fromarray(disc_region_img)

        if not os.path.exists(data_save_path +'/disc/'):
            os.makedirs(data_save_path +'/disc/')
        disc_region_img.save(data_save_path +'/disc/' + temp_txt[0][:-4] + '.png')

        run_time_start = time()

        temp_img = pro_process(disc_region, CDRSeg_size)
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        temp_img = imagenet_utils.preprocess_input(temp_img.astype(np.float32), mode='tf')
        prob = CDRSeg_model.predict(temp_img)

        run_time_end = time()
        print(' Run time MNet: ' + str(run_time_end - run_time_start) + '   Img number: ' + str(lineIdx + 1))

        prob_map = np.squeeze(prob)
        img_name = temp_txt[0][:-4] + '.png'
        if dataset == "refuge":
            img_name = temp_txt[0][:-4] + '.bmp'

        save_per_img(org_img, disc_region, mask_path=mask_path,
                 data_save_path=data_save_path, img_name=img_name, prob_map=prob_map, err_coord=err_coord,
                 crop_coord=crop_coord, DiscROI_size=DiscROI_size,
                 org_img_size=org_img.shape, threshold=0.75, pt=False, ext=save_data_type)
    evaluate_segmentation_results(data_save_path, mask_path,ext='png', output_path='./')
    print(load_from)
    print(data_save_path)
