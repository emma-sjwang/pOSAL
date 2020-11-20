import csv
import numpy as np

from scipy.io import savemat
from os import listdir, path, makedirs


def parse_boolean(input_string):
    '''
    Parse a string as a boolean
    '''
    return input_string.upper()=='TRUE'



def get_filenames(path_to_files, extension):
    '''
    Get all the files on a given folder with the given extension
    Input:
        path_to_files: string to a path where the files are
        [extension]: string representing the extension of the files
    Output:
        image_filenames: a list of strings with the filenames in the folder
    '''

    # initialize a list of image filenames
    image_filenames = []
    # add to this list only those filenames with the corresponding extension
    for file in listdir(path_to_files):
        if file.endswith('.' + extension):
            image_filenames = image_filenames + [ file ]

    return image_filenames



def read_csv_classification_results(csv_filename):
    '''
    Read a two-column CSV file that has the classification results inside.
    Input:
        csv_filename: full path and filename to a two column CSV file with the classification results (image filename, score)
    Output:
        image_filenames: list of image filenames, as retrieved from the first column of the CSV file
        scores: numpy array of floats, as retrieved from the second column of the CSV file
    '''

    # initialize the output variables
    image_filenames = []
    scores = []

    # open the file
    with open(csv_filename, 'r') as csv_file:
        # initialize a reader
        csv_reader = csv.reader(csv_file)
        # ignore the first row, that only has the header
        next(csv_reader)
        # and now, iterate and fill the arrays
        for row in csv_reader:
            image_filenames = image_filenames + [ row[0] ]
            scores = scores + [ float(row[1]) ]

    # turn the list of scores into a numpy array
    scores = np.asarray(scores, dtype=np.float)

    # return the image filenames and the scores
    return image_filenames, scores



def sort_scores_by_filename(target_names, names_to_sort, values_to_sort):
    '''
    This function is intended to correct the ordering in the outputs, just in case...
    Input:
        target_names: a list of names sorted in the order that we want
        names_to_sort: a list of names to sort
        values_to_sort: a numpy array of values to sort
    Output:
        sorted_values: same array than values_to_sort, but this time sorted :)
    '''

    names_to_sort = [x.upper() for x in names_to_sort]

    # initialize the array of sorted values
    sorted_values = np.zeros(values_to_sort.shape)

    # iterate for each filename in the target names
    for i in range(len(target_names)):
        # assign the value to the correct position in the array
        sorted_values[i] = values_to_sort[names_to_sort.index(target_names[i].upper())]
    
    # return the sorted values
    return sorted_values



def sort_coordinates_by_filename(target_names, names_to_sort, values_to_sort):
    '''
    This function is intended to correct the ordering in the outputs, just in case...
    Input:
        target_names: a list of names sorted in the order that we want
        names_to_sort: a list of names to sort
        values_to_sort: a numpy array of values to sort
    Output:
        sorted_values: same array than values_to_sort, but this time sorted :)
    '''

    # initialize the array of sorted values
    sorted_values = np.zeros(values_to_sort.shape)

    # iterate for each filename in the target names
    for i in range(len(target_names)):
        # assign the value to the correct position in the array
        sorted_values[i,:] = values_to_sort[names_to_sort.index(target_names[i])]
    
    # return the sorted values
    return sorted_values



def get_labels_from_training_data(gt_folder):
    '''
    Since the training data has two folder, "Glaucoma" and "Non-Glaucoma", we can use
    this function to generate an array of labels automatically, according to the image
    filenames
    Input:
        gt_folder: path to the training folder, with "Glaucoma" and "Non-Glaucoma" folder inside
    Output:
        image_filenames: filenames in the gt folders
        labels: binary labels (0: healthy, 1:glaucomatous)
    '''

    # prepare the folders to read
    glaucoma_folder = path.join(gt_folder, 'Glaucoma')
    non_glaucoma_folder = path.join(gt_folder, 'Non-Glaucoma')

    # get all the filenames inside each folder
    glaucoma_filenames = get_filenames(glaucoma_folder, 'bmp')
    non_glaucoma_filenames = get_filenames(non_glaucoma_folder, 'bmp')

    # concatenate them to generate the array of image filenames
    image_filenames = glaucoma_filenames + non_glaucoma_filenames

    # generate the array of labels
    labels = np.zeros(len(image_filenames), dtype=np.bool)
    labels[0:len(glaucoma_filenames)] = True

    return image_filenames, labels



def save_roc_curve(filename, tpr, fpr, auc):
    '''
    Save the ROC curve values on a .mat file
    Input:
        filename: output filename
        tpr: true positive rate
        fpr: false positive rate
        auc: area under the ROC curve
    '''

    # save the current ROC curve as a .mat file for MATLAB
    savemat(filename, {'tpr': tpr, 'fpr' : fpr, 'auc': auc})



def save_csv_classification_performance(output_filename, auc, reference_sensitivity):
    '''
    Save the AUC and the reference sensitivity values in a CSV file
    Input:
        output_filename: a string with the full path and the output file name (with .csv extension)
        auc: area under the ROC curve
        reference_sensitivity: sensitivity value for a given specificity
    '''

    # open the file
    with open(output_filename, 'w') as csv_file:
        # initialize the writer
        my_writer = csv.writer(csv_file)
        # write the column names
        my_writer.writerow(['AUC', 'Sensitivity'])
        # write the values
        my_writer.writerow([str(auc), str(reference_sensitivity)])



def save_csv_fovea_location_performance(output_filename, distance):
    '''
    Save the mean Euclidean distance on a CSV file
    Input:
        output_filename: a string with the full path and the output file name (with .csv extension)
        distance: mean Euclidean distance
    '''

    # open the file
    with open(output_filename, 'w') as csv_file:
        # initialize the writer
        my_writer = csv.writer(csv_file)
        # write the column names
        my_writer.writerow(['Mean Euclidean distance'])
        # write the values
        my_writer.writerow([str(distance)])



def save_csv_segmentation_table(table_filename, image_filenames, cup_dices, disc_dices, ae_cdrs):
    '''
    Save the table of segmentation results as a CSV file.
    Input:
        table_filename: a string with the full path and the table filename (with .csv extension)
        image_filenames: a list of strings with the names of the images
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    '''

    # write the data
    with open(table_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Filename', 'Cup-Dice', 'Disc-Dice', 'AE-CDR'])
        # write each row
        for i in range(len(image_filenames)):
            table_writer.writerow( [image_filenames[i], str(cup_dices[i]), str(disc_dices[i]), str(ae_cdrs[i])] )



def save_csv_fovea_location_table(table_filename, image_filenames, distances):
    '''
    Save the table of Euclidean distances results as a CSV file.
    Input:
        table_filename: a string with the full path and the table filename (with .csv extension)
        image_filenames: a list of strings with the names of the images
        distances: a 1D numpy array with the Euclidean distances of the prediction, for each image
    '''

    # write the data
    with open(table_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Filename', 'Euclidean distance'])
        # write each row
        for i in range(len(image_filenames)):
            table_writer.writerow( [image_filenames[i], str(distances[i])] )



def save_csv_mean_segmentation_performance(output_filename, mean_cup_dice, mean_disc_dice, mae_cdrs):
    '''
    Save a CSV file with the mean performance
    Input:
        output_filename: a string with the full path and the table filename (with .csv extension)
        mean_cup_dice: average Dice coefficient for the optic cups
        mean_disc_dice: average Dice coefficient for the optic discs
        mae_cdrs: mean absolute error of the vertical cup to disc ratios
    '''

    # write the data
    with open(output_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Cup-Dice', 'Disc-Dice', 'AE-CDR'])
        # write each row
        table_writer.writerow( [ str(mean_cup_dice), str(mean_disc_dice), str(mae_cdrs)] )



def read_fovea_location_results(csv_filename):
    '''
    Read a CSV file with 3 columns: the first contains the filenames, and the second/third have
    the (x,y) coordinates, respectively.
    Input:
        csv_filename: full path and filename to a three columns CSV file with the fovea location results (image filename, x, y)
    Output:
        image_filenames: list of image filenames, as retrieved from the first column of the CSV file
        coordinates: a 2D numpy array of coordinates
    '''

    # initialize the output variables
    image_filenames = []
    coordinates = None

    # open the file
    with open(csv_filename, 'r') as csv_file:
        # initialize a reader
        csv_reader = csv.reader(csv_file)
        # ignore the first row, that only has the header
        next(csv_reader)
        # and now, iterate and fill the arrays
        for row in csv_reader:
            # append the filename
            image_filenames = image_filenames + [ row[0] ]
            # append the coordinates
            current_coordinates = np.asarray( row[1:], dtype=np.float )
            if coordinates is None:
                coordinates = current_coordinates
            else:
                coordinates = np.vstack( (coordinates, current_coordinates))

    return image_filenames, coordinates



# import openpyxl

# def read_gt_fovea_location(xlsx_filename, is_training=False):
#     '''
#     Read a XLSX file with 3 columns: the first contains the filenames, and the second/third have
#     the (x,y) coordinates, respectively.
#     Input:
#         xlsx_filename: full path and filename to a three columns XLSX file with the fovea location results (image filename, x, y)
#         [is_training]: boolean indicating if we are using training data or no
#     Output:
#         image_filenames: list of image filenames, as retrieved from the first column of the CSV file
#         coordinates: a 2D numpy array of coordinates
#     '''

#     # initialize the output variables
#     image_filenames = []
#     coordinates = None

#     # read the xlsx file
#     book = openpyxl.load_workbook(xlsx_filename)
#     current_sheet = book.active

#     # iterate for each row
#     for row in current_sheet.iter_rows(min_row=2, min_col=1):
#         # append the filename
#         image_filenames = image_filenames + [ row[1].value ]
#         # append the coordinates
#         if is_training:
#             current_coordinates = np.asarray( [ float(row[2].value), float(row[3].value) ], dtype=np.float )
#         else:
#             current_coordinates = np.asarray( [ float(row[3].value), float(row[4].value) ], dtype=np.float )
#         if coordinates is None:
#             coordinates = current_coordinates
#         else:
#             coordinates = np.vstack( (coordinates, current_coordinates))

#     return image_filenames, coordinates



# import openpyxl

# def read_gt_labels(xlsx_filename):
#     '''
#     Read a XLSX file with 2 columns: the first contains the filenames, and the second/third have
#     the binary label for glaucoma (1) / healthy (0).
#     Input:
#         xlsx_filename: full path and filename to a three columns XLSX file with the fovea location results (image filename, x, y)
#     Output:
#         image_filenames: list of image filenames, as retrieved from the first column of the CSV file
#         labels: a 2D numpy array of coordinates
#     '''

#     # initialize the output variables
#     image_filenames = []
#     labels = None

#     # read the xlsx file
#     book = openpyxl.load_workbook(xlsx_filename)
#     current_sheet = book.active

#     # iterate for each row
#     for row in current_sheet.iter_rows(min_row=2, min_col=1):
#         # append the filename
#         current_name = row[0].value[:-3]  + 'jpg'
#         image_filenames = image_filenames + [ current_name ]
#         # append the coordinates
#         current_label = row[1].value > 0
#         if labels is None:
#             labels = current_label
#         else:
#             labels = np.vstack( (labels, current_label))

#     return image_filenames, labels



import zipfile

def unzip_submission(submission_file, output_folder):
    '''
    Unzip a .ZIP file with a submission to REFUGE from a team
    Input:
        submission_file: full path and filename of the .zip file
        output_folder: folder where the output will be saved
    '''

    # initialize the output folder
    if not path.exists(output_folder):
        makedirs(output_folder)

    # open the zip file
    zip_ref = zipfile.ZipFile(submission_file, 'r')
    zip_ref.extractall(output_folder)
    zip_ref.close()



def export_table_of_results(table_filename, team_names, segmentation_results, classification_results, fovea_detection_results):
    '''
    Export a table of results (unsorted) as a CSV
    Input:
        table_filename: filename of the CSV file with the table of results
        team_names: names of the teams evaluated
        segmentation_results: list of segmentation results
        classification_results: list of classification results
        fovea_detection_results: list of fovea detection results
    '''

    # write the data
    with open(table_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Team name', 'Mean optic cup Dice', 'Mean optic disc Dice', 'MAE cup to disc ratio', 'AUC', 'Reference Sensitivity', 'Mean Euclidean distance'])
        # write each row
        for i in range(len(team_names)):
            # retrieve current results
            current_segmentation_results = segmentation_results[i]
            current_classification_results = classification_results[i]
            current_fovea_detection_results = fovea_detection_results[i]
            # write a row of results
            table_writer.writerow( [team_names[i], str(current_segmentation_results[0]), str(current_segmentation_results[1]), str(current_segmentation_results[2]),
                                    str(current_classification_results[0]), str(current_classification_results[1]), str(current_fovea_detection_results)] )



def export_ranking(table_filename, header, team_names, scores):
    '''
    Export the ranking
    Input:
        table_filename: filename of the CSV file with the table of results
        header: list of strings with the header for the output file
        team_names: names of the teams evaluated
        scores: a numpy array with ranking information
    '''

    scores = np.asarray(scores)

    # write the data
    with open(table_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(header)
        # write each row
        for i in range(len(team_names)):
            # write a row of results
            if len(scores.shape) > 1:
                table_writer.writerow( [ team_names[i] ] + scores[i,:].tolist() )
            else:
                table_writer.writerow( [ team_names[i] ] + [ scores[i] ] )



def read_table_of_results(table_filename):
    '''
    Read the table of results (unsorted) as a CSV
    Input:
        table_filename: filename of the CSV file with the table of results
    Output:
        header: a list of strings with the name of the evaluation metrics
        teams: a list of strings with the name of the teams
        results: a numpy matrix of evaluation metrics
    '''

    # open the file
    with open(table_filename, 'r') as csv_file:
        # initialize the reader
        csv_reader = csv.reader(csv_file)
        # get the first row
        header = next(csv_reader)[1:]

        # initialize the list of teams
        teams = []
        # initialize a numpy matrix with all the other results
        results = None
        # and now, iterate and fill the arrays
        for row in csv_reader:
            # append the team name
            teams = teams + [ row[0] ]
            # append the results
            current_results = np.asarray( row[1:], dtype=np.float )
            if results is None:
                results = current_results
            else:
                results = np.vstack( (results, current_results))

    return header, teams, results
