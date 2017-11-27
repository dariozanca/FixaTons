'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena

@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072

@date: September, 2017
'''

#########################################################################################

import os
COLLECTION_PATH = os.path.dirname(os.path.abspath(__file__)) + '/FixaTons'

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import cv2
import numpy as np

import _get_data_functions as get

#########################################################################################

def map(DATASET_NAME, STIMULUS_NAME,

              showSalMap=True,
              showFixMap=False,

              wait_time=5000,

              plotMaxDim=0):

    ''' This functions uses cv2 standard library to visualize a specified
        stimulus.

        By default, stimulus is shown with its saliency map aside. It is possible
        to deactivate such option by setting the additional argument showSalMap=False.

        It is possible to show also (or alternatively) the fixation map by setting
        the additional argument showFixMap=True.

        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''

    stimulus = get.stimulus(DATASET_NAME, STIMULUS_NAME)

    toPlot = stimulus

    if showSalMap:
        saliency_map = get.saliency_map(DATASET_NAME, STIMULUS_NAME)
        saliency_map = cv2.cvtColor(saliency_map,cv2.COLOR_GRAY2RGB)
        toPlot = np.concatenate((toPlot,saliency_map), axis=1)

    if showFixMap:
        fixation_map = get.fixation_map(DATASET_NAME, STIMULUS_NAME)
        fixation_map = cv2.cvtColor(fixation_map,cv2.COLOR_GRAY2RGB)*255
        toPlot = np.concatenate((toPlot,fixation_map), axis=1)

    if plotMaxDim:
            h, w, _ = np.shape(toPlot)
            h, w = float(h), float(w)
            if h > w:
                w = (plotMaxDim / h) * w
                h = plotMaxDim
            else:
                h = (plotMaxDim / w) * h
                w = plotMaxDim
            h, w = int(h), int(w)
            toPlot = cv2.resize(toPlot, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imshow(STIMULUS_NAME, toPlot)

    cv2.waitKey(wait_time)

    cv2.destroyAllWindows()

    return

#########################################################################################

def scanpath(DATASET_NAME, STIMULUS_NAME, subject = 0, 
                       animation = True, wait_time = 5000,
                       putLines = True, putNumbers = True, 
                       plotMaxDim = 1024):

    ''' This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.

        By default, one random scanpath is chosen between available subjects. For
        a specific subject, it is possible to specify its id on the additional
        argument subject=id.

        It is possible to visualize it as an animation by setting the additional
        argument animation=True.

        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''

    stimulus = get.stimulus(DATASET_NAME, STIMULUS_NAME)

    scanpath = get.scanpath(DATASET_NAME, STIMULUS_NAME, subject)

    toPlot = [stimulus,] # look, it is a list!

    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        cv2.circle(frame,
                   (fixation[0], fixation[1]),
                   10, (0, 0, 255), 3)
        if putNumbers:
            cv2.putText(frame, str(i+1),
                        (fixation[0], fixation[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,0), thickness=2)
        if putLines and i>0:
            prec_fixation = scanpath[i-1].astype(int)
            cv2.line(frame, 
						(prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]), 
						(0, 0, 255), thickness = 1, lineType = 8, shift = 0)

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation: toPlot.pop(0)

    # if required, resize the frames
    if plotMaxDim:
        for i in range(len(toPlot)):
            h, w, _ = np.shape(toPlot[i])
            h, w = float(h), float(w)
            if h > w:
                w = (plotMaxDim / h) * w
                h = plotMaxDim
            else:
                h = (plotMaxDim / w) * h
                w = plotMaxDim
            h, w = int(h), int(w)
            toPlot[i] = cv2.resize(toPlot[i], (w, h), interpolation=cv2.INTER_CUBIC)

    for i in range(len(toPlot)):

        cv2.imshow('Scanpath of '+subject+' watching '+STIMULUS_NAME,
                   toPlot[i])
        if i == 0:
            milliseconds = 1
        elif i == 1:
            milliseconds = scanpath[0,3]
        else:
            milliseconds = scanpath[i-1,3] - scanpath[i-2,2]
        milliseconds *= 1000

        cv2.waitKey(int(milliseconds))

    cv2.waitKey(wait_time)

    cv2.destroyAllWindows()



    return
