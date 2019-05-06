'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

from Model.models import *
from Utils.data_generator import *
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import *

import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def change_learning_rate(model, base_lr, iter, max_iter, power):
    new_lr = lr_poly(base_lr, iter, max_iter, power)
    K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)


def change_learning_rate_D(model, base_lr, iter, max_iter, power):
    new_lr = lr_poly(base_lr, iter, max_iter, power)
    K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':

    ''' parameter setting '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    DiscROI_size = 512
    CDRSeg_size = 512
    lr = 2.5e-5
    LEARNING_RATE_D = 1e-5
    batch_size = 4
    dataset_t = "Drishti-GS/"

    dataset = "refuge/"
    total_num = 400
    total_epoch = 100
    total_epoch_stop = total_epoch / 2
    power = 0.9

    if dataset == "refuge/":
        total_num = 320

    weights_path = "weights/" + dataset_t + "/DA_patch/_{epoch:04d}.hdf5"
    load_from = "./weights/weights1.h5"

    weights_root = os.path.dirname(weights_path)
    G_weights_root = os.path.join(weights_root, 'Generator')
    D_weights_root = os.path.join(weights_root, 'Discriminator')

    if not os.path.exists(G_weights_root):
        print("Create save weights folder on %s\n\n" % weights_root)
        os.makedirs(G_weights_root)
        os.makedirs(D_weights_root)
    _MODEL = os.path.basename(__file__).split('.')[0]

    logs_path = "./log_tf/" + dataset_t + "/DA_patch/"
    logswriter = tf.summary.FileWriter
    print("logtf path: %s \n\n" % logs_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    summary_writer = logswriter(logs_path)

    ''' define model '''
    model_Generator = Model_CupSeg(input_shape = (CDRSeg_size, CDRSeg_size, 3), classes=2, backbone='mobilenetv2', lr=lr)

    model_Discriminator = Discriminator(input_shape=(CDRSeg_size, CDRSeg_size, 2),
                                                  learning_rate=LEARNING_RATE_D)
    model_Adversarial = Sequential()
    model_Discriminator.trainable = False
    model_Adversarial.add(model_Generator)
    model_Adversarial.add(model_Discriminator)
    model_Adversarial.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')

    # model_Generator.summary()
    # plot_model(model_Generator, to_file='deeplabv3.png')

    if os.path.exists(load_from):
        print('Loading weight for generator model from file {}\n\n'.format(load_from))
        model_Generator.load_weights(load_from)
    else:
        print('[ERROR:] CANNOT find weight file {}\n\n'.format(load_from))

    ''' define data generator '''
    # train0 means 4/5 training data from REFUGE dataset
    trainGenerator_Gene = Generator_Gene(batch_size, './data/'+dataset+'/train0/disc_small', DiscROI_size,
                                         CDRSeg_size = CDRSeg_size, pt=False, phase='train')
    if dataset_t in ['refuge']: # using val data to val
        valGenerator_Gene = Generator_Gene(batch_size, './data/' + dataset_t + '/val/disc_small', DiscROI_size,
                                           CDRSeg_size=CDRSeg_size, pt=False, phase='val')
    else: # for other dataset using test data to val
        valGenerator_Gene = Generator_Gene(batch_size, './data/'+dataset_t+'/test/disc_small', DiscROI_size,
                                           CDRSeg_size = CDRSeg_size, pt=False, phase='val')
    if dataset_t in ['refuge']: # using val data to train without ground truth
        trainAdversarial_Gene = Adversarial_Gene(batch_size, './data/' + dataset_t + '/val/disc_small', DiscROI_size,
                                                 CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    else: # using train split of target domain to train without ground truth
        trainAdversarial_Gene = Adversarial_Gene(batch_size, './data/' + dataset_t + '/train/disc_small', DiscROI_size,
                                             CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    trainDS_Gene = GD_Gene(batch_size, './data/' + dataset + '/train0/disc_small', True,
                           CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    if dataset_t in ['refuge']:
        trainDT_Gene = GD_Gene(batch_size,
                               './data/' + dataset_t + '/val/disc_small', False,
                               CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    else:
        trainDT_Gene = GD_Gene(batch_size,
                           './data/' + dataset_t + '/train/disc_small', False,
                           CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)

    ''' train for epoch and iter one by one '''
    epoch = 0
    dice_loss_val = 0
    disc_coef_val = 0
    cup_coef_val = 0
    results_eva = [0, 0, 0]
    results_DS = 0
    results_DT = 0
    for epoch in range(total_epoch):
        loss = 0
        smooth_loss = 0
        dice_loss = 0
        disc_coef = 0
        cup_coef = 0

        loss_A = 0
        loss_GD = 0
        loss_DS = 0
        loss_DT = 0
        loss_A_map = 0
        loss_A_scale = 0
        loss_DS_map = 0
        loss_DS_scale = 0
        loss_DT_map = 0
        loss_DT_scale = 0
        results_A = 0

        iters_total = int(total_num/batch_size + 1)
        for iter in range(iters_total):

            ''' train Generator '''
            # source domain
            img_S, mask_S = next(trainGenerator_Gene)
            results_G = model_Generator.train_on_batch(img_S, mask_S)

            loss += results_G[0]/iters_total
            disc_coef += results_G[1]/iters_total
            cup_coef += results_G[2]/iters_total
            smooth_loss += results_G[3]/iters_total
            dice_loss += results_G[4]/iters_total

            # target domain
            img_T, output_T = next(trainAdversarial_Gene)
            results_A = model_Adversarial.train_on_batch(img_T, output_T)

            loss_A += np.array(results_A) / iters_total

            # print log information every 10 iterations
            if (iter + 1) % 10 == 0:
                img, mask = next(valGenerator_Gene)
                results_eva = model_Generator.evaluate(img, mask)
                dice_loss_val += results_eva[0] / (iters_total/20)
                disc_coef_val += results_eva[1] / (iters_total/20)
                cup_coef_val += results_eva[2] / (iters_total/20)
                print('[EVALUATION: (iter: {})]\n{}:{},{}:{},{}:{}' \
                      .format(iter+1, model_Generator.metrics_names[0],results_eva[0],
                                                 model_Generator.metrics_names[1],results_eva[1],
                                                 model_Generator.metrics_names[2], results_eva[2]))

            ''' train Discriminator '''
            img, label = next(trainDS_Gene)
            prediction = model_Generator.predict(img)
            results_DS = model_Discriminator.train_on_batch(prediction, label)
            loss_DS += results_DS / iters_total

            img, label = next(trainDT_Gene)
            prediction = model_Generator.predict(img)
            results_DT = model_Discriminator.train_on_batch(prediction, label)
            loss_DT += results_DT / iters_total

            ''' visulization through tensorboard '''
            summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag='loss', simple_value=float(results_G[0])),
                tf.Summary.Value(
                    tag='disc_coef', simple_value=float(results_G[1])),
                tf.Summary.Value(
                    tag='cup_coef', simple_value=float(results_G[2])),
                tf.Summary.Value(
                    tag='smooth_loss', simple_value=float(results_G[3])),
                tf.Summary.Value(
                    tag='dice_loss', simple_value=float(results_G[4])),
                tf.Summary.Value(
                    tag='loss_A', simple_value=float(results_A)),
                tf.Summary.Value(
                    tag='loss_DS', simple_value=float(results_DS)),
                tf.Summary.Value(
                    tag='loss_DT', simple_value=float(results_DT)),
                tf.Summary.Value(
                    tag='loss_val', simple_value=float(results_eva[0])),
                tf.Summary.Value(
                    tag='disc_coef_val', simple_value=float(results_eva[1])),
                tf.Summary.Value(
                    tag='cup_coef_val', simple_value=float(results_eva[2])),
            ])
            summary_writer.add_summary(summary, epoch*iters_total + iter)

        ''' show logs every epoch'''
        print('\n\nepoch = {0:8d}, dice_loss = {1:.3f}, disc_coef = {2:.4f}, cup_coef = {3:.4f}, learning_rate={4}'.format(
            epoch, dice_loss, disc_coef, cup_coef, K.get_value(model_Generator.optimizer.lr)))

        ''' save model weight every 10 epochs'''
        if (epoch+1) % 10 == 0:
            G_weights_path = os.path.join(G_weights_root, 'generator_%s.h5' % ( epoch + 1 ))
            D_weights_path = os.path.join(D_weights_root, 'discriminator_%s.h5' % ( epoch + 1 ))
            print("Save model to %s" % G_weights_path)
            model_Generator.save_weights(G_weights_path, overwrite=True)
            print("Save model to %s" % D_weights_path)
            model_Discriminator.save_weights(D_weights_path, overwrite=True)

        # update learning rate
        change_learning_rate(model_Generator, lr, epoch, total_epoch, power)
        change_learning_rate(model_Adversarial, lr, epoch, total_epoch, power)
        change_learning_rate_D(model_Discriminator, LEARNING_RATE_D, epoch, total_epoch, power)
