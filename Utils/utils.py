'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from skimage import measure,draw
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    img = scipy.misc.imresize(img, (input_size, input_size, 3))
    return img


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def transfer_mask2maps(mask):
    mask = mask[:,:,0]
    disc_mask = (mask > 250).astype(np.uint8)
    cup_mask = (mask > 120).astype(np.uint8)
    return disc_mask, cup_mask


def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def save_img(org_img, mask_path, data_save_path, img_name, prob_map, err_coord, crop_coord, DiscROI_size, org_img_size, threshold=0.5, pt=False):
    path = os.path.join(data_save_path, img_name)
    path0 = os.path.join(data_save_path+'/visulization/', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))

    disc_map = resize(prob_map[:, :, 0], (DiscROI_size, DiscROI_size))
    cup_map = resize(prob_map[:, :, 1], (DiscROI_size, DiscROI_size))

    if pt:
        disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                          DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                         DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    mask_path = os.path.join(mask_path, img_name.split('.')[0]+'.bmp')
    if os.path.exists(mask_path):
        fig, (ax1,ax2,ax3, ax4) = plt.subplots(nrows=1, ncols=4)  # create figure & 1 axis
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # ax0, ax1, ax2 = axes.ravel()
    ax1.imshow(org_img)
    ax2.imshow(org_img)
    ax3.imshow(org_img)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('initial image', fontsize=4, color='b')
    # plt.figure(dpi=600)

    disc_mask = (disc_map > threshold) # return binary mask
    cup_mask = (cup_map > threshold)
    disc_mask = disc_mask.astype(np.uint8)
    cup_mask = cup_mask.astype(np.uint8)

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    for n, contour in enumerate(contours_disc):
        ax2.plot(contour[:, 1] + crop_coord[2] - err_coord[2], contour[:, 0] + crop_coord[0] - err_coord[0], 'b', linewidth=0.2)
    for n, contour in enumerate(contours_cup):
        ax2.plot(contour[:, 1] + crop_coord[2] - err_coord[2], contour[:, 0] + crop_coord[0] - err_coord[0], 'g', linewidth=0.2)

    for i in range(5):
        disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)

    disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

    disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8) # return 0,1
    cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
    ROI_result = disc_mask + cup_mask
    ROI_result[ROI_result < 1] = 255
    ROI_result[ROI_result < 2] = 128
    ROI_result[ROI_result < 3] = 0

    Img_result = np.zeros((org_img_size[0], org_img_size[1], 3), dtype=int) + 255
    Img_result[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], 0] = ROI_result[err_coord[0]:err_coord[1],
                                                                              err_coord[2]:err_coord[3]]
    # for submit
    cv2.imwrite(path, Img_result[:, :, 0])

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    for n, contour in enumerate(contours_disc):
        ax3.plot(contour[:, 1] + crop_coord[2] - err_coord[2], contour[:, 0] + crop_coord[0] - err_coord[0], 'b',
                 linewidth=0.2)
    for n, contour in enumerate(contours_cup):
        ax3.plot(contour[:, 1] + crop_coord[2] - err_coord[2], contour[:, 0] + crop_coord[0] - err_coord[0], 'g',
                 linewidth=0.2)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('raw image', fontsize=4, color='b')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('smooth image', fontsize=4, color='b')

    if os.path.exists(mask_path):
        ax4.imshow(org_img)
        mask = np.asarray(image.load_img(mask_path))
        disc_mask, cup_mask = transfer_mask2maps(mask)
        print('onon3')
        contours_disc = measure.find_contours(disc_mask, 0.5)
        contours_cup = measure.find_contours(cup_mask, 0.5)
        for n, contour in enumerate(contours_disc):
            ax4.plot(contour[:, 1], contour[:, 0], 'b',
                     linewidth=0.2)
        for n, contour in enumerate(contours_cup):
            ax4.plot(contour[:, 1], contour[:, 0], 'g',
                     linewidth=0.2)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('ground truth', fontsize=4, color='b')
        fig.savefig(path0, dpi=600, bbox_inches='tight')  # save the figure to file
        plt.close(fig)
    else:
        fig.savefig(path0, dpi=600, bbox_inches='tight')  # save the figure to file
        plt.close(fig)


def dice_coef(groundtruth, prediction):
    '''
    :param groundtruth: [batchsize, H, W]
    :param prediction: [batchsize, H, W]
    :return: 1 scalars, dice coeffience
    '''
    prediction[prediction>=0.5] = 1
    prediction[prediction<0.5] = 0
    axis = tuple(range(1, len(prediction.shape)))
    intersection = np.sum(groundtruth * prediction, axis=axis)
    return np.mean((intersection + 1e-8) / (np.sum(groundtruth, axis=axis) + np.sum(prediction, axis=axis) - intersection + 1e-8))


def calculate_dice(results, groundtruth):
    '''
    :param results: [batchsize, H, W, channels]
    :param groundtruth: [batchsize, H, W, channels]
    :return: output: 2 scalars, for disc and cup
    '''

    return dice_coef(groundtruth[:,:,:, 1], results[:,:,:, 1]), dice_coef(groundtruth[:,:,:, 0], results[:,:,:, 0])


def disc_crop(org_img, DiscROI_size, C_x, C_y, fill_value=0):
    tmp_size = int(DiscROI_size / 2)
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    if fill_value != 0:
        disc_region = disc_region + fill_value
    crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
    err_coord = [0, DiscROI_size, 0, DiscROI_size]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3], ]

    return disc_region, err_coord, crop_coord
