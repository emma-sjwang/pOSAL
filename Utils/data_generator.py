'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''
# from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from skimage.transform import rotate
from keras.preprocessing import image
import scipy
from PIL import Image
from skimage.measure import label, regionprops
from keras.applications import imagenet_utils
from matplotlib.pyplot import imshow, imsave
import math
from Utils.random_eraser import get_random_eraser
import random
from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from skimage.exposure import adjust_log, equalize_adapthist
from skimage.filters.rank import median
from skimage.morphology import disk
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


eraser = get_random_eraser()


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    img = scipy.misc.imresize(img, (input_size, input_size, 3))
    return img


def elastic_transform(image, label, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    seed = random.random()
    if seed > 0.5:
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        transformed_image = np.zeros(image.shape)
        transformed_label = np.zeros(image.shape)
        for i in range(image.shape[-1]):
            transformed_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape)
            if label is not None:
                transformed_label[:, :, i] = map_coordinates(label[:, :, i], indices, order=1).reshape(shape)
            else:
                transformed_label = None
        transformed_image = transformed_image.astype(np.uint8)
        if label is not None:
            transformed_label = transformed_label.astype(np.uint8)
        return transformed_image, transformed_label
    else:
        return image, label


def add_salt_pepper_noise(image):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = image.copy()
    row, col, _ = image.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))

    seed = random.random()
    if seed > 0.75:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
        X_imgs_copy[coords[0], coords[1], :] = 1
    elif seed > 0.5:
        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
        X_imgs_copy[coords[0], coords[1], :] = 0
    else:
        return image
    return X_imgs_copy


def adjust_light(image):
    seed = random.random()
    if seed > 0.5:
        gamma = random.random() * 3 + 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image.astype(np.uint8), table).astype(np.uint8)
    return image


def adjust_contrast(image):
    seed = random.random()
    if seed > 0.5:
        image = adjust_log(image, 1)
    return image


def ad_blur(image):
    seed = random.random()
    if seed > 0.5:
        image = image / 127.5 - 1
        image[:,:,0] = median(image[:,:,0], disk(3))
        image[:,:,1] = median(image[:,:,1], disk(3))
        image[:,:,2] = median(image[:,:,2], disk(3))
        image = (image + 1) * 127.5
    image = image.astype(np.uint8)
    return image


def data_augmentation(img, mask=None, flip=True, ifrotate=True, ifelastic=True, scale=True, noise=True, light=True, erasing=True, ad_contrast=False, clahe=False, blur=False):

    if mask is not None:

        if scale:
            seed = random.random()
            if seed > 0.5:
                img_ = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
                mask_ = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])
                scalar = random.random() + 0.5
                new_size = int(img.shape[0] * scalar)
                if new_size % 2 != 0:
                    new_size += 1
                img = scipy.misc.imresize(img.astype('float32'), [new_size, new_size, 3])
                mask = scipy.misc.imresize(mask.astype('float32'), [new_size, new_size, 3])

                if new_size < img_.shape[0]:
                    range = img_.shape[0] - new_size
                    x_shift = random.randint(0, range-1)
                    y_shift = random.randint(0, range-1)
                    img_[x_shift:x_shift+new_size, y_shift:y_shift+new_size,:] = img
                    mask_[x_shift:x_shift+new_size, y_shift:y_shift+new_size,:] = mask
                else:

                    range = new_size - img_.shape[0]
                    if range < 1:
                        img_ = img.astype('float32')
                    else:
                        x_shift = random.randint(0, range-1)
                        y_shift = random.randint(0, range-1)
                        img_ = img[x_shift:x_shift+img_.shape[0], y_shift:y_shift+img_.shape[1], :]
                        mask_ = mask[x_shift:x_shift+img_.shape[0], y_shift:y_shift+img_.shape[1], :]

                img = img_
                mask = mask_
        if ifrotate:
            seed = random.random()
            if seed > 0.5:
                angle = random.randint(1, 4) * 90
                img = rotate(img / 255.0, angle) * 255
                mask = rotate(mask / 255.0, angle) * 255
                img = img.astype(np.uint8)
                mask = mask.astype(np.uint8)

        if flip:
            seed = random.random()
            if seed > 0.5:
                img = img[:,::-1,:]
                mask = mask[:,::-1]
            seed = random.random()
            if seed > 0.5:
                img = img[::-1, :, :]
                mask = mask[::-1, :]

    else:

        if scale:
            seed = random.random()
            if seed > 0.5:
                img_ = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)
                scalar = random.random() + 0.5
                new_size = int(img.shape[0] * scalar)
                if new_size % 2 != 0:
                    new_size += 1
                img = scipy.misc.imresize(img.astype('float32'), [new_size, new_size, 3])
                if new_size < img_.shape[0]:
                    range = img_.shape[0] - new_size
                    x_shift = random.randint(0, range - 1)
                    y_shift = random.randint(0, range - 1)
                    img_[x_shift:x_shift + new_size, y_shift:y_shift + new_size, :] = img
                else:
                    range = new_size - img_.shape[0]
                    if range == 0:
                        img_ = img
                    else:
                        x_shift = random.randint(0, range - 1)
                        y_shift = random.randint(0, range - 1)
                        img_ = img[x_shift:x_shift + img_.shape[0], y_shift:y_shift + img_.shape[1], :]

                img = img_

        if flip:
            seed = random.random()
            if seed > 0.5:
                img = img[:,::-1,:]
            seed = random.random()
            if seed > 0.5:
                img = img[::-1, :, :]

        if ifrotate:
            seed = random.random()
            if seed > 0.5:
                angle = random.randint(1, 4) * 90
                img = rotate(img / 255.0, angle) * 255
                img = img.astype(np.uint8)



    if ifelastic:
        img, mask = elastic_transform(img, mask, img.shape[1] * 2, img.shape[1] * 0.08)
    if ad_contrast:
        img = adjust_contrast(img)
    if blur:
        img = ad_blur(img)
    if noise:
        img = add_salt_pepper_noise(img)
    if light:
        img = adjust_light(img)
    if erasing:
        img = eraser(img)
    return img, mask


def polar_transform(img, mask, img_size, mode='train'):
    random_rotate_seed = 3 * 90
    width_shift = 0
    heigth_shift = 0

    if mode == 'train':
        random_rotate_seed = random.randint(0,3) * 90
        width_shift = random.randint(0,40) - 20
        heigth_shift = random.randint(0,40) - 20
    img = rotate(cv2.linearPolar(img, (img_size / 2 + width_shift, img_size / 2 + heigth_shift), img_size / 2 - 20,
                                 cv2.WARP_FILL_OUTLIERS), random_rotate_seed)

    mask = rotate(cv2.linearPolar(mask, (img_size / 2 + width_shift, img_size / 2 + heigth_shift), img_size / 2 - 20,
                                  cv2.WARP_FILL_OUTLIERS), random_rotate_seed)

    return img, mask


def train_generator(batch_size, train_path, img_size, CDRSeg_size = 400, pt=False):
    train_ids = next(os.walk(train_path + '/image/'))[2]
    X = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.uint8)
    for i, id_ in enumerate(train_ids):
        path = os.path.join(train_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        X[i] = img

        mask_ = np.asarray(
            image.load_img(os.path.join(train_path, 'mask/', id_.split('.')[0] + ".png"), grayscale=True))

        Y[i,:,:,0] = mask_

    index_list = list(range(0, len(train_ids)))
    random.shuffle(index_list)
    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.uint8)
    Y_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 2), dtype=np.uint8)
    index = 0
    while True:

        batch_index = 0
        while batch_index < batch_size:
            img = X[index_list[index]]
            mask_ = Y[index_list[index], :,:,0]

            mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            mask[mask_ < 200] = [255, 0, 0]
            mask[mask_ < 50] = [255, 255, 0]
            img, mask = data_augmentation(img.copy(), mask.copy())

            img = pro_process(img, CDRSeg_size)
            mask = pro_process(mask, CDRSeg_size)
            X_train[batch_index] = img
            Y_train[batch_index] = mask[:,:,0:2]/255.0

            batch_index += 1
            index += 1
            if index == len(train_ids):
                index = 0
                random.shuffle(index_list)
        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield X_train, Y_train


def val_generator(batch_size, test_path, img_size, CDRSeg_size = 400, pt=False):
    test_ids = next(os.walk(test_path + '/image/'))[2]

    X = np.zeros((len(test_ids), CDRSeg_size, CDRSeg_size, 3), dtype=np.uint8)
    Y = np.zeros((len(test_ids), CDRSeg_size, CDRSeg_size, 3), dtype=np.uint8)
    for i, id_ in enumerate(test_ids):
        path = os.path.join(test_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        mask_ = np.asarray(
            image.load_img(os.path.join(test_path, 'mask/', id_.split('.')[0] + ".png"), grayscale=True))
        mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        mask[mask_ < 200] = [255, 0, 0]
        mask[mask_ < 50] = [255, 255, 0]
        if pt:
            img, mask = polar_transform(img.copy(), mask.copy(), img_size, CDRSeg_size, mode='val')
        img = pro_process(img, CDRSeg_size)
        mask = pro_process(mask, CDRSeg_size)
        X[i] = img
        Y[i] = mask/255.0

    index_list = list(range(0, len(test_ids)))
    random.shuffle(index_list)
    X_test = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.uint8)
    Y_test = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 2), dtype=np.uint8)
    index = 0
    while True:
        batch_index = 0
        while batch_index < batch_size:
            X_test[batch_index] = X[index_list[index]]
            Y_test[batch_index] = Y[index_list[index]][:,:,0:2]
            batch_index += 1
            index += 1
            if index == len(test_ids):
                index = 0
                random.shuffle(index_list)
        X_test = imagenet_utils.preprocess_input(X_test.astype(np.float32), mode='tf')
        yield X_test, Y_test


def cls_generator(batch_size, train_path, img_size, input_shape = (224, 224, 5), phase='train', noise_label=True):
    train_ids = next(os.walk(train_path + '/image/'))[2]
    data = {'0':[], '1':[]}
    for i, id_ in enumerate(train_ids):
        path = os.path.join(train_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        img = pro_process(img, input_shape[0])

        if id_[0] =='g':
            data['1'].append(img)
        else:
            data['0'].append(img)

    index_list = list(range(0, len(train_ids)))
    random.shuffle(index_list)
    X_train = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]), dtype='float32')
    Y_train = np.zeros((batch_size, 1), dtype='float32')
    index = 0
    while True:
        batch_index = 0
        while batch_index < batch_size:
            seed = random.random()
            img, mask_ = None, None
            LABEL = 0
            if seed < 0.5:
                LABEL = 0
                seed_index = random.randint(0, len(data['0'])-1)
                img = data['0'][seed_index]
            else:
                LABEL = 1
                seed_index = random.randint(0, len(data['1']) - 1)
                img = data['1'][seed_index]
            if phase == 'train':
                img, mask = data_augmentation(img.copy())
                if noise_label:
                    seed = random.random()
                    if seed < 0.05:
                        LABEL = 1 - LABEL

            img = imagenet_utils.preprocess_input(img.astype(np.float32), mode='tf')
            X_train[batch_index, :, :, 0:3] = img
            Y_train[batch_index, :] = LABEL

            batch_index += 1
            index += 1
            if index == len(train_ids):
                index = 0
                random.shuffle(index_list)

        yield X_train, Y_train


def Generator_Gene(batch_size, train_path, img_size, CDRSeg_size=400, pt=False, phase='train'):
    train_ids = next(os.walk(train_path + '/image/'))[2]
    X = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.uint8)
    for i, id_ in enumerate(train_ids):
        path = os.path.join(train_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        X[i] = img
        mask_ = np.asarray(
            image.load_img(os.path.join(train_path, 'mask/', id_.split('.')[0] + ".png"), grayscale=True))
        Y[i, :, :, 0] = mask_

    index_list = list(range(0, len(train_ids)))
    random.shuffle(index_list)
    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.float32)
    Y_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 2), dtype=np.float32)
    index = 0
    while True:
        batch_index = 0
        while batch_index < batch_size:
            img = X[index_list[index]]
            mask_ = Y[index_list[index], :, :, 0]

            mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            mask[mask_ < 200] = [255, 0, 0]
            mask[mask_ < 50] = [255, 255, 0]

            if phase == 'train':
                img, mask = data_augmentation(img.copy(), mask.copy())

            img = pro_process(img, CDRSeg_size)
            mask = pro_process(mask, CDRSeg_size)

            X_train[batch_index] = img
            Y_train[batch_index] = mask[:, :, 0:2] / 255.0

            batch_index += 1
            index += 1
            if index == len(train_ids):
                index = 0
                random.shuffle(index_list)
        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield X_train, Y_train


def SmoothAndNoiseLabel(label, noise_label=True, smooth_label=True, threshold=0.05):
    if noise_label:
        seed = random.random()
        if seed < threshold:
            label = 1 - label
    if smooth_label:
        seed = random.random() * 0.5 - 0.25
        smoothlabel = label + seed
    else:
        smoothlabel = label
    if smoothlabel < 0:
        smoothlabel = 0

    return smoothlabel


def Adversarial_Gene(batch_size, target_path, img_size, CDRSeg_size=400, phase='train', noise_label=True):
    train_ids = next(os.walk(target_path + '/image/'))[2]
    X = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)

    for i, id_ in enumerate(train_ids):
        path = os.path.join(target_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        X[i] = img

    index_list = list(range(0, len(train_ids)))
    random.shuffle(index_list)
    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.float32)

    output1_size = int(CDRSeg_size)
    Y2_train = np.zeros((batch_size, 16,16,1), dtype=np.float32)

    index = 0

    while True:
        label = 1.0
        batch_index = 0
        smoothlabel = 1.0
        while batch_index < batch_size:
            img = X[index_list[index]]

            if phase == 'train':
                img, mask = data_augmentation(img.copy())
                if noise_label:
                    smoothlabel = SmoothAndNoiseLabel(label)

            img = pro_process(img, CDRSeg_size)

            X_train[batch_index] = img
            Y2_train[batch_index,:,:, 0] = np.zeros([16, 16]) + smoothlabel

            batch_index += 1
            index += 1
            if index == len(train_ids):
                index = 0
                random.shuffle(index_list)
        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield (X_train, Y2_train)

def Adversarial_Gene_single(batch_size, target_path, img_size, CDRSeg_size=400, phase='train', noise_label=True, scale=True):
    train_ids = next(os.walk(target_path + '/image/'))[2]
    X = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)

    for i, id_ in enumerate(train_ids):
        path = os.path.join(target_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        X[i] = img

    index_list = list(range(0, len(train_ids)))
    random.shuffle(index_list)
    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.float32)
    if scale:
        output_size = int(1)
        Y_train = np.zeros((batch_size, 1), dtype=np.float32)
    else:
        output_size = int(CDRSeg_size)
        Y_train = np.zeros((batch_size, output_size, output_size, 1), dtype=np.float32)

    index = 0

    while True:
        label = 1.0
        batch_index = 0
        smoothlabel = 1.0
        while batch_index < batch_size:
            img = X[index_list[index]]

            if phase == 'train':
                img, mask = data_augmentation(img.copy())
                if noise_label:
                    smoothlabel = SmoothAndNoiseLabel(label)

            img = pro_process(img, CDRSeg_size)

            X_train[batch_index] = img
            if scale:
                Y_train[batch_index, 0] = smoothlabel
            else:
                Y_train[batch_index, :, :, 0] = np.zeros([output_size, output_size]) + smoothlabel

            batch_index += 1
            index += 1
            if index == len(train_ids):
                index = 0
                random.shuffle(index_list)
        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield X_train, Y_train


def GD_Gene(batch_size, source_path, source=True, CDRSeg_size=400, phase='train', noise_label=True, smooth_label=True):
    data = []
    source_ids = next(os.walk(source_path + '/image/'))[2]
    for i, id_ in enumerate(source_ids):
        path = os.path.join(source_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        data.append(img)

    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.float32)
    Y2_train = np.zeros((batch_size, 16,16,1), dtype=np.float32)

    if source:
        label = 0
    else:
        label = 1

    while True:
        batch_index = 0
        smoothlabel = float(label)
        while batch_index < batch_size:
            seed = random.randint(0, len(data) - 1)
            img = data[seed]

            if phase == 'train':
                img, mask = data_augmentation(img.copy())
                if noise_label:
                    smoothlabel = SmoothAndNoiseLabel(label)
            img = pro_process(img, CDRSeg_size)

            X_train[batch_index] = img
            Y2_train[batch_index, :,:,0] = np.zeros([16, 16]) +  smoothlabel

            batch_index += 1

        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield (X_train, Y2_train)


def GD_Gene_single(batch_size, source_path, source=True, CDRSeg_size=400, phase='train', noise_label=True, smooth_label=True, scale=True):
    data = []

    source_ids = next(os.walk(source_path + '/image/'))[2]
    for i, id_ in enumerate(source_ids):
        path = os.path.join(source_path, 'image/', id_)
        img = np.asarray(image.load_img(path))
        data.append(img)

    X_train = np.zeros((batch_size, CDRSeg_size, CDRSeg_size, 3), dtype=np.float32)
    if scale:
        output_size = int(1)
        Y_train = np.zeros((batch_size, 1), dtype=np.float32)
    else:
        output_size = int(CDRSeg_size)
        Y_train = np.zeros((batch_size, output_size, output_size, 1), dtype=np.float32)

    if source:
        label = 0
    else:
        label =1
    while True:
        batch_index = 0
        smoothlabel = float(label)
        while batch_index < batch_size:
            seed = random.randint(0, len(data) - 1)
            img = data[seed]

            if phase == 'train':
                img, mask = data_augmentation(img.copy())
                if noise_label:
                    smoothlabel = SmoothAndNoiseLabel(label)
            img = pro_process(img, CDRSeg_size)

            X_train[batch_index] = img
            if scale:
                Y_train[batch_index, 0] = smoothlabel
            else:
                Y_train[batch_index, :, :, 0] = np.zeros([output_size, output_size]) + smoothlabel

            batch_index += 1

        X_train = imagenet_utils.preprocess_input(X_train.astype(np.float32), mode='tf')
        yield X_train, Y_train

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


def save_img(mask_path, data_save_path, img_name, prob_map, err_coord, crop_coord, DiscROI_size, org_img_size, threshold=0.5, pt=False):
    path = os.path.join(data_save_path, img_name)

    disc_map = resize(prob_map[:, :, 0], (DiscROI_size, DiscROI_size))
    cup_map = resize(prob_map[:, :, 1], (DiscROI_size, DiscROI_size))

    if pt:
        disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                          DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                         DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    for i in range(5):
        disc_map = scipy.signal.medfilt2d(disc_map, 7)
        cup_map = scipy.signal.medfilt2d(cup_map, 7)

    disc_mask = (disc_map > threshold) # return binary mask
    cup_mask = (cup_map > threshold)

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
    cv2.imwrite(path, Img_result[:,:,0])

    if os.path.exists(mask_path):
        mask_path = os.path.join(mask_path, img_name)
        mask = np.asarray(image.load_img(mask_path))
        gt_mask_path = os.path.join(data_save_path, 'gt_mask', img_name)
        gt_mask = 0.5 * mask + 0.5 * Img_result
        gt_mask = gt_mask.astype(np.uint8)
        imsave(gt_mask_path, gt_mask)


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
