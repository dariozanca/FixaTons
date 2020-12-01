'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena

@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072

@date: September, 2017
'''

#########################################################################################

from FixaTons import COLLECTION_PATH

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import os
import cv2
import numpy as np
import random

#########################################################################################

def stimulus(DATASET_NAME, STIMULUS_NAME):

    ''' This functions returns the matrix of pixels of a specified stimulus.

        Notice that, of course, both DATASET_NAME and STIMULUS_NAME need
        to be specified. The latter, must include file extension.

        The returned matrix could be 2- or 3-dimesional. '''

    return cv2.imread(
        os.path.join(
            COLLECTION_PATH,
            DATASET_NAME,
            'STIMULI',
            STIMULUS_NAME
        ), 1)

#########################################################################################

def fixation_map(DATASET_NAME, STIMULUS_NAME):

    ''' This functions returns the matrix of pixels of the fixation map
        of a specified stimulus.

        Notice that, of course, both DATASET_NAME and STIMULUS_NAME need
        to be specified. The latter, must include file extension.

        The returned matrix is a 2-dimesional matrix with 1 on fixated
        locations and 0 elsewhere. '''

    file_name, _ = os.path.splitext(STIMULUS_NAME)

    # File extensions of the fixation maps may be different between
    # datasets, so we need a check at this point.
    # All the files in the same folder have the same extension.

    _, file_extension = os.path.splitext(
        os.listdir(
            os.path.join(
                COLLECTION_PATH,
                DATASET_NAME,
                'FIXATION_MAPS')
        )[0])

    # Get the matrix

    fixation_map = cv2.imread(
        os.path.join(COLLECTION_PATH,
                     DATASET_NAME,
                     'FIXATION_MAPS',
                     file_name+file_extension
                     ), 1)

    fixation_map = fixation_map[:, :, 0]

    fixation_map[fixation_map > 0] = 1

    return fixation_map

#########################################################################################

def saliency_map(DATASET_NAME, STIMULUS_NAME):

    ''' This functions returns the matrix of pixels of the saliency map
        of a specified stimulus. Saliency map has been obtained by convolving
        the fixation map with a proper gaussian filter (corresponding to
        one degree of visual angle).

        Notice that, of course, both DATASET_NAME and STIMULUS_NAME need
        to be specified. The latter, must include file extension.

        The returned matrix is a 2-dimesional matrix. Values are in the
        range [0,1]. '''

    file_name, _ = os.path.splitext(STIMULUS_NAME)

    # File extensions of the saliency maps may be different between
    # datasets, so we need a check at this point.
    # All the files in the same folder have the same extension.

    _, file_extension = os.path.splitext(
                            os.listdir(
                                os.path.join(
                                    COLLECTION_PATH,
                                    DATASET_NAME,
                                    'SALIENCY_MAPS')
                            )[0]
    )

    # Get the matrix
    return cv2.imread(
        os.path.join(
            COLLECTION_PATH,
            DATASET_NAME,
            'SALIENCY_MAPS',
            file_name + file_extension
        ), 0)

#########################################################################################

def scanpath(DATASET_NAME, STIMULUS_NAME,

                 subject = 0):

    ''' This functions returns the matrix of fixations of a specified stimulus. The
        scanpath matrix contains a row for each fixation. Each row is of the type
        [x, y, initial_t, final_time]

        By default, one random scanpath is chosen between available subjects. For
        a specific subject, it is possible to specify its id on the additional
        argument subject=id. '''

    file_name, _ = os.path.splitext(STIMULUS_NAME)

    if not subject:
        list_of_subjects = os.listdir(
            os.path.join(
                COLLECTION_PATH,
                DATASET_NAME,
                'SCANPATHS',
                file_name)
        )
        subject = random.choice(list_of_subjects)

    scanpath_file = open(
        os.path.join(
            COLLECTION_PATH,
            DATASET_NAME,
            'SCANPATHS',
            file_name,
            subject), 'r')

    scanpath_file_lines = scanpath_file.readlines()

    scanpath = np.zeros((len(scanpath_file_lines), 4))

    for i in range(len(scanpath)):
        scanpath[i] = np.array(scanpath_file_lines[i].split()).astype(np.float)

    return scanpath