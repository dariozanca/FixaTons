'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena

@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072

@date: October, 2017
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import FixaTons
import os
import numpy as np

#########################################################################################

COLLECTION_PATH = os.path.dirname(os.path.abspath(__file__)) + '/FixaTons'

#########################################################################################

def statistics(DATASET_NAME=None):

    ''' This function compute statistics on Fixations collection.

        If a DATASET_NAME is specified, statistics computation is restricted to that
        dataset. '''

    number_of_scanpaths = 0
    fixations_per_second = 0
    saccade_length = 0

    if DATASET_NAME:
        datasets_list = (DATASET_NAME,)
    else:
        datasets_list = FixaTons.list.datasets()

    for DATASET_NAME in datasets_list:
        for STIMULUS_NAME in FixaTons.list.stimuli(DATASET_NAME):
            for SUBJECT_NAME in FixaTons.list.subjects(DATASET_NAME, STIMULUS_NAME):

                number_of_scanpaths += 1

                scanpath = FixaTons.get.scanpath(
                                DATASET_NAME, STIMULUS_NAME, SUBJECT_NAME)

                if not len(scanpath) == 0:
                    fixations_per_second += fps(scanpath)
                    saccade_length += sac_len(scanpath)
                else:
                    number_of_scanpaths -= 1

    # average it
    fixations_per_second /= number_of_scanpaths
    saccade_length /= number_of_scanpaths

    return fixations_per_second, saccade_length

def fps(scanpath):

    if len(scanpath) == 0:
        return 0
    else:
        return float(len(scanpath))/scanpath[-1,3]

def sac_len(scanpath):

    if len(scanpath) == 0:
        return 0
    else:

        sum = 0

        for i in np.arange(1,len(scanpath), 1):

            sum += np.sqrt(
                        (scanpath[i, 0] - scanpath[i-1, 0])**2 +
                        (scanpath[i, 1] - scanpath[i - 1, 1]) ** 2
            )

        return sum / len(scanpath)