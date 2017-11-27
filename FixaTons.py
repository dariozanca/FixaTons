'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena

@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072

@date: September, 2017
'''

#########################################################################################

import os
COLLECTION_NAME = os.path.dirname(os.path.abspath(__file__)) + '/FixaTons'

'''
This file includes tools to an easy use of the collection of datasets. 
This tools help you in different tasks:
    - List information
    - Get data (matrices)
    - Visualize data
    - Compute metrics
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import os
import cv2
import numpy as np

import _list_information_functions as list
import _get_data_functions as get
import _visualize_data_functions as show
import _visual_attention_metrics as metrics
import _compute_statistics as stats