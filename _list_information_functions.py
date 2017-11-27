'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena

@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072

@date: September, 2017
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import os

#########################################################################################

COLLECTION_PATH = os.path.dirname(os.path.abspath(__file__)) + '/FixaTons'

#########################################################################################

def datasets():

    ''' This functions lists the names of the datasets included in the collection '''

    return os.listdir(COLLECTION_PATH)

#########################################################################################

def stimuli(DATASET_NAME):

    ''' This functions lists the names of the stimuli of a specified dataset '''

    return os.listdir(COLLECTION_PATH+'/'+DATASET_NAME+'/STIMULI')

#########################################################################################

def subjects(DATASET_NAME, STIMULUS_NAME):

    ''' This functions lists the names of the subjects which have been watching a
        specified stimuli of a dataset '''

    file_name, _ = os.path.splitext(STIMULUS_NAME)

    return os.listdir(COLLECTION_PATH+'/'+DATASET_NAME+'/SCANPATHS/'+file_name)