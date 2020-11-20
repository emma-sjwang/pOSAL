'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

import numpy as np

from scipy import misc
from os import path, makedirs

from Utils.file_management import get_filenames, save_csv_mean_segmentation_performance#, save_csv_segmentation_table



EPS = 1e-7


def dice_coefficient(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = 2 * intersection / (segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value



def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    Input:
        binary_segmentation: a boolean 2D numpy array representing a region of interest.
    Output:
        diameter: the vertical diameter of the structure, defined as the largest diameter between the upper and the lower interfaces
    '''

    # turn the variable to boolean, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter)

    # return it
    return float(diameter)



def vertical_cup_to_disc_ratio(segmentation):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    The vertical cup to disc ratio is defined as here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1722393/pdf/v082p01118.pdf
    Input:
        segmentation: binary 2D numpy array representing a segmentation, with 0: optic cup, 128: optic disc, 255: elsewhere.
    Output:
        cdr: vertical cup to disc ratio
    '''

    # compute the cup diameter
    cup_diameter = vertical_diameter(segmentation==0)
    # compute the disc diameter
    disc_diameter = vertical_diameter(segmentation<255)

    return cup_diameter / (disc_diameter + EPS)
def evaluate_binary_segmentation(segmentation, gt_label):
    '''
    Compute the evaluation metrics of the REFUGE challenge by comparing the segmentation with the ground truth
    Input:
        segmentation: binary 2D numpy array representing the segmentation, with 0: optic cup, 128: optic disc, 255: elsewhere.
        gt_label: binary 2D numpy array representing the ground truth annotation, with the same format
    Output:
        cup_dice: Dice coefficient for the optic cup
        disc_dice: Dice coefficient for the optic disc
        cdr: absolute error between the vertical cup to disc ratio as estimated from the segmentation vs. the gt_label, in pixels
    '''

    # compute the Dice coefficient for the optic cup
    cup_dice = dice_coefficient(segmentation==0, gt_label==0)
    # compute the Dice coefficient for the optic disc
    disc_dice = dice_coefficient(segmentation<255, gt_label<255)
    # compute the absolute error between the cup to disc ratio estimated from the segmentation vs. the gt label
    cdr = absolute_error(vertical_cup_to_disc_ratio(segmentation), vertical_cup_to_disc_ratio(gt_label))

    return cup_dice, disc_dice, cdr



def absolute_error(predicted, reference):
    '''
    Compute the absolute error between a predicted and a reference outcomes.
    Input:
        predicted: a float value representing a predicted outcome
        reference: a float value representing the reference outcome
    Output:
        abs_err: the absolute difference between predicted and reference
    '''

    return abs(predicted - reference)



def generate_table_of_results(image_filenames, segmentation_folder, gt_folder, is_training=False):
    '''
    Generates a table with image_filename, cup_dice, disc_dice and cdr values
    Input:
        image_filenames: a list of strings with the names of the images.
        segmentation_folder: a string representing the full path to the folder where the segmentation files are
        gt_folder: a string representing the full path to the folder where the ground truth annotation files are
        is_training: a boolean value indicating if the evaluation is performed on training data or not
    Output:
        image_filenames: same as the input parameter
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    '''

    # initialize an array for the Dice coefficients of the optic cups
    cup_dices = np.zeros(len(image_filenames), dtype=np.float)
    # initialize an array for the Dice coefficients of the optic discs
    disc_dices = np.zeros(len(image_filenames), dtype=np.float)
    # initialize an array for the absolute errors of the vertical cup to disc ratios
    ae_cdrs = np.zeros(len(image_filenames), dtype=np.float)

    # iterate for each image filename
    for i in range(len(image_filenames)):

        # read the segmentation
        segmentation = misc.imread(path.join(segmentation_folder, image_filenames[i]))
        if len(segmentation.shape) > 2:
            segmentation = segmentation[:,:,0]
        # read the gt
        if is_training:
            gt_filename = path.join(gt_folder, 'Glaucoma', image_filenames[i])
            if path.exists(gt_filename):
                gt_label = misc.imread(gt_filename)
            else:
                gt_filename = path.join(gt_folder, 'Non-Glaucoma', image_filenames[i])
                if path.exists(gt_filename):
                    gt_label = misc.imread(gt_filename)
                else:
                    raise ValueError('Unable to find {} in your training folder. Make sure that you have the folder organized as provided in our website.'.format(image_filenames[i]))
        else:
            gt_filename = path.join(gt_folder, image_filenames[i])
            if path.exists(gt_filename):
                gt_label = misc.imread(gt_filename)
            else:
                raise ValueError('Unable to find {} in your ground truth folder. If you are using training data, make sure to use the parameter is_training in True.'.format(image_filenames[i]))

        # evaluate the results and assign to the corresponding row in the table
        cup_dices[i], disc_dices[i], ae_cdrs[i] = evaluate_binary_segmentation(segmentation, gt_label)

    # return the colums of the table
    return image_filenames, cup_dices, disc_dices, ae_cdrs



def get_mean_values_from_table(cup_dices, disc_dices, ae_cdrs):
    '''
    Compute the mean evaluation metrics for the segmentation task.
    Input:
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    Output:
        mean_cup_dice: the mean Dice coefficient for the optic cups
        mean_disc_dice: the mean Dice coefficient for the optic disc
        mae_cdr: the mean absolute error for the vertical cup to disc ratio
    '''

    # compute the mean values of each column
    mean_cup_dice = np.mean(cup_dices)
    mean_disc_dice = np.mean(disc_dices)
    mae_cdr = np.mean(ae_cdrs)
    
    return mean_cup_dice, mean_disc_dice, mae_cdr


def evaluate_segmentation_results(segmentation_folder, gt_folder, ext='bmp', output_path=None):
    '''
    Evaluate the segmentation results of a single submission
    Input:
        segmentation_folder: full path to the segmentation files
        gt_folder: full path to the ground truth files
        [output_path]: a folder where the results will be saved. If not provided, the results are not saved
        [export_table]: a boolean value indicating if the table will be exported or not
        [is_training]: a boolean value indicating if the evaluation is performed on training data or not
    Output:
        mean_cup_dice: the mean Dice coefficient for the optic cups
        mean_disc_dice: the mean Dice coefficient for the optic disc
        mae_cdr: the mean absolute error for the vertical cup to disc ratio
    '''

    # get all the image filenames
    image_filenames = get_filenames(segmentation_folder, ext)
    if len(image_filenames)==0:
        print('** The segmentation folder does not include any bmp file. Check the files extension and resubmit your results.')
        raise ValueError()
    # create output path if it does not exist
    if not (output_path is None) and not (path.exists(output_path)):
        makedirs(output_path)

    # generate a table of results
    _, cup_dices, disc_dices, ae_cdrs = generate_table_of_results(image_filenames, segmentation_folder, gt_folder, False)

    # compute the mean values
    mean_cup_dice, mean_disc_dice, mae_cdr = get_mean_values_from_table(cup_dices, disc_dices, ae_cdrs)
    # print the results on screen
    print('Dice Optic Cup = {}\nDice Optic Disc = {}\nMAE CDR = {}'.format(str(mean_cup_dice), str(mean_disc_dice), str(mae_cdr)))
    # save the mean values in the output path
    if not(output_path is None):
        # initialize the output filename
        output_filename = path.join(output_path, 'evaluation_segmentation.csv')
        # save the results
        save_csv_mean_segmentation_performance(output_filename, mean_cup_dice, mean_disc_dice, mae_cdr)

    # return the average performance
    return mean_cup_dice, mean_disc_dice, mae_cdr