'''
@author: Dario Zanca, Ph.D.
@institutions: University of Siena

@e-mail: dariozanca@gmail.it
@tel: (+39) 333 82 78 072

@date: October, 2017
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import os

#########################################################################################

from FixaTons import COLLECTION_PATH

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

    return os.listdir(
        os.path.join(
            COLLECTION_PATH,
            DATASET_NAME,
            'SCANPATHS',
            file_name
        )
    )