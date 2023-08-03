'''
Created on 1 mar 2017

@author: 	Dario Zanca 
@summary: 	Collection of functions to compute visual attention metrics for:
                - saliency maps similarity
                    - AUC Judd (Area Under the ROC Curve, Judd version)
                    - KL Kullback Leiber divergence
                    - NSS Normalized Scanpath Similarity
                - scanpaths similarity
                    - Euclidean distance
                    - String-edit distance
                    - Scaled time-delay embeddings
                    - String-based time-delay embeddings
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance

#########################################################################################

##############################  saliency metrics  #######################################

#########################################################################################

''' created: Tilke Judd, Oct 2009
    updated: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This measures how well the saliencyMap of an image predicts the ground truth human 
fixations on the image. ROC curve created by sweeping through threshold values determined
by range of saliency map values at fixation locations;
true positive (tp) rate correspond to the ratio of saliency map values above threshold 
at fixation locations to the total number of fixation locations, false positive (fp) rate
correspond to the ratio of saliency map values above threshold at all other locations to 
the total number of posible other locations (non-fixated image pixels) '''


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score


######################################################################################

def AUC_shuffled(saliencyMap, fixationMap, otherMap, Nsplits, stepSize=0.1, toPlot=False):
    '''saliencyMap is the saliency map
    fixationMap is the human fixation map (binary matrix)
    otherMap is a binary fixation map (like fixationMap) by taking the union of
    fixations from M other random images (Borji uses M=10)
    Nsplits is number of random splits
    stepSize is for sweeping through saliency map
    if toPlot=1, displays ROC curve
    '''

    # saliencyMap = saliencyMap.transpose()
    # fixationMap = fixationMap.transpose()
    # otherMap = otherMap.transpose()

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        saliencyMap = np.array(Image.fromarray(saliencyMap).resize((np.shape(fixationMap)[1], np.shape(fixationMap)[0])))

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    S = saliencyMap.flatten(order='F')
    F = fixationMap.flatten(order='F')
    Oth = otherMap.flatten(order='F')

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations specified by otherMap
    ind = np.nonzero(Oth)[0] # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.empty((Nfixations_oth,Nsplits))
    randfix[:] = np.nan

    for i in range(Nsplits):
        randind = ind[np.random.permutation(len(ind))]  # randomize choice of fixation locations
        randfix[:, i] = S[randind[:Nfixations_oth]] # sal map values at random fixation locations of other random images

    # calculate AUC per random split (set of random locations)
    auc = np.empty(Nsplits)
    auc[:] = np.nan

    def Matlab_like_gen(start, stop, step, precision):
        r = start
        while round(r, precision) <= stop:
            yield round(r, precision)
            r += step

    for s in range(Nsplits):
        curfix = randfix[:, s]
        i0 = Matlab_like_gen(0, max(np.maximum(Sth, curfix)), stepSize, 5)
        allthreshes = [x for x in i0]
        allthreshes.reverse()

        tp = np.zeros((len(allthreshes) + 2))
        fp = np.zeros((len(allthreshes) + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i+1] = (Sth >= thresh).sum() / Nfixations
            fp[i+1] = (curfix >= thresh).sum() / Nfixations_oth

        auc[s] = np.trapz(tp, x=fp)

    score = np.mean(auc)  # mean across random splits

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This finds the KL-divergence between two different saliency maps when viewed as 
distributions: it is a non-symmetric measure of the information lost when saliencyMap 
is used to estimate fixationMap. '''


def KLdiv(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map

    # convert to float
    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)

    # make sure maps have the same shape
    from scipy.misc import imresize
    map1 = imresize(map1, np.shape(map2))

    # make sure map1 and map2 sum to 1
    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    # compute KL-divergence
    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This finds the normalized scanpath saliency (NSS) between two different saliency maps. 
NSS is the average of the response values at human eye positions in a model saliency 
map that has been normalized to have zero mean and unit standard deviation. '''


def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make sure maps have the same shape
    from scipy.misc import imresize
    map1 = imresize(saliencyMap, np.shape(fixationMap))
    if not map1.max() == 0:
        map1 = map1.astype(float) / map1.max()

    # normalize saliency map
    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1)

    # mean value at fixation locations
    score = map1[fixationMap.astype(bool)].mean()

    return score


######################################################################################

'''created: Zoya Bylinskii, March 6
based on: Kummerer et al.
(http://www.pnas.org/content/112/52/16054.abstract)
Python Implementation: Lapo Faggi, Oct 2020

This finds the information-gain of the saliencyMap over a baselineMap'''


def InfoGain(saliencyMap, fixationMap, baselineMap):
    '''saliencyMap is the saliency map
    fixationMap is the human fixation map (binary matrix)
    baselineMap is another saliency map (e.g. all fixations from other images)'''

    map1 = np.resize(saliencyMap,np.shape(fixationMap))
    mapb = np.resize(baselineMap, np.shape(fixationMap))

    # normalize and vectorize saliency maps
    map1 = (map1.flatten(order='F') - np.min(map1))/ (np.max(map1 - np.min(map1)))
    mapb = (mapb.flatten(order='F') - np.min(mapb))/(np.max(mapb - np.min(mapb)))

    # turn into distributions
    map1 /= np.sum(map1)
    mapb /= np.sum(mapb)

    fixationMap = fixationMap.flatten(order = 'F')
    locs = fixationMap > 0

    eps = 2.2204e-16
    score = np.mean(np.log2(eps+map1[locs])-np.log2(eps+mapb[locs]))

    return score


#########################################################################################

##############################  scanpaths metrics  ######################################

#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the Euclidean distance between two scanpath of the same length. '''


def euclidean_distance(human_scanpath, simulated_scanpath):
    if len(human_scanpath) == len(simulated_scanpath):

        dist = np.zeros(len(human_scanpath))
        for i in range(len(human_scanpath)):
            P = human_scanpath[i]
            Q = simulated_scanpath[i]
            dist[i] = np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
        return dist

    else:

        print('Error: The two sequences must have the same length!')
        return False


#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the string edit distance metric.
    
    Given an image, it is divided in nxn regions. To each region, a letter is assigned. 
    For each scanpath, the correspondent letter is assigned to each fixation, depending 
    the region in which such fixation falls. So that each scanpath is associated to a 
    string. 
    
    Distance between the two generated string is then compared as described in 
    "speech and language processing", Jurafsky, Martin. Cap. 3, par. 11. '''



def scanpath_to_string(
        scanpath,
        stimulus_width,
        stimulus_height,
        grid_size
):
    width_step = stimulus_width // grid_size
    height_step = stimulus_height // grid_size

    string = ''

    for i in range(np.shape(scanpath)[0]):
        fixation = scanpath[i].astype(np.int32)
        correspondent_square = (fixation[0] // width_step) + (fixation[1] // height_step) * grid_size
        string += chr(97 + int(correspondent_square))

    return string

def string_edit_distance(
        scanpath_1,
        scanpath_2,
        stimulus_width,
        stimulus_height,
        grid_size=4
):

    string_1 = scanpath_to_string(
        scanpath_1,
        stimulus_width,
        stimulus_height,
        grid_size
    )
    string_2 = scanpath_to_string(
        scanpath_2,
        stimulus_width,
        stimulus_height,
        grid_size
    )

    return edit_distance(string_1, string_2, transpositions=True), string_1, string_2



#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the metric described in "Simulating Human Saccadic 
    Scanpaths on Natural Images", by Wei Wang, Cheng Chen, Yizhou Wang, 
    Tingting Jiang, Fang Fang, Yuan Yao 

    Time-delay embedding are used in order to quantitatively compare the 
    stochastic and dynamic scanpaths of varied lengths '''


def time_delay_embedding_distance(
        human_scanpath,
        simulated_scanpath,

        # options
        k=3,  # time-embedding vector dimension
        distance_mode='Mean'
        ):
    # human_scanpath and simulated_scanpath can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(human_scanpath) < k or len(simulated_scanpath) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    human_scanpath_vectors = []
    for i in np.arange(0, len(human_scanpath) - k + 1):
        human_scanpath_vectors.append(human_scanpath[i:i + k])

    simulated_scanpath_vectors = []
    for i in np.arange(0, len(simulated_scanpath) - k + 1):
        simulated_scanpath_vectors.append(simulated_scanpath[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s_k_vec in simulated_scanpath_vectors:

        # find human k-vec of minimum distance

        norms = []

        for h_k_vec in human_scanpath_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False


def scaled_time_delay_embedding_distance(
        human_scanpath,
        simulated_scanpath,
        image,

        # options
        toPlot=False):
    # to preserve data, we work on copies of the lists
    H_scanpath = copy(human_scanpath)
    S_scanpath = copy(simulated_scanpath)

    # First, coordinates are rescaled as to an image with maximum dimension 1
    # This is because, clearly, smaller images would produce smaller distances
    max_dim = float(max(np.shape(image)))

    for P in H_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    for P in S_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    # Then, scanpath similarity is computer for all possible k
    max_k = min(len(H_scanpath), len(S_scanpath))
    similarities = []
    for k in np.arange(1, max_k + 1):
        s = time_delay_embedding_distance(
            H_scanpath,
            S_scanpath,
            k=k,  # time-embedding vector dimension
            distance_mode='Mean')
        similarities.append(np.exp(-s))
        print(similarities[-1])

    # Now that we have similarity measure for all possible k
    # we compute and return the mean

    if toPlot:
        keys = np.arange(1, max_k + 1)
        plt.plot(keys, similarities)
        plt.show()

    return sum(similarities) / len(similarities)




#########################################################################################

''' created: Dario Zanca, April 2022

    First, we converst scanpaths to strings. Then, we compute a distance with  time-delay 
    embeddings in the string domain. '''


def string_based_time_delay_embedding_distance(
        scanpath_1,
        scanpath_2,

        stimulus_width,
        stimulus_height,

        # options
        k=3,  # time-embedding vector dimension
        distance_mode='Mean'
        ):
    # human_scanpath and simulated_scanpath can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(scanpath_1) < k or len(scanpath_2) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    scanpath_1_vectors = []
    for i in np.arange(0, len(scanpath_1) - k + 1):
        scanpath_1_vectors.append(scanpath_1[i:i + k])

    scanpath_2_vectors = []
    for i in np.arange(0, len(scanpath_2) - k + 1):
        scanpath_2_vectors.append(scanpath_2[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s2_k_vec in scanpath_2_vectors:

        # find human k-vec of minimum distance

        norms = []

        for s1_k_vec in scanpath_1_vectors:

            d, _, _ = string_edit_distance(s1_k_vec, s2_k_vec, stimulus_width=stimulus_width, stimulus_height=stimulus_height)
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False


#########################################################################################

''' Implementation of recurrence quantification analysis (RQA) to quantify the similarity of two fixation sequences (or scanpaths).
    Based on the paper Anderson, Nicola C., et al. "A comparison of scanpath comparison methods." Behavior research methods 47.4 (2015): 1377-1392.
    It implements the following metrics:

    Determinism
    Laminarity
    Center of recurrence mass
    Cross-recurrence'''


from _rqa import RQA

def compute_rqa_metrics(scanpath_1, scanpath_2, r=60, ll=2):
    '''Parameters:
        scanpath_1            (n,2) matrix of x,y fixations
        scanpath_2            (n,2) matrix of x,y fixations
        r                     e.g. 60, radius threshold
        ll                    e.g. 2, minimum line length
        
        Returns:    
        det                   Determinism
        lam                   Laminarity
        corm                  Center of recurrence mass
        crossrec              Cross-recurrence'''

    if scanpath_1.shape[1] > 2:
        scanpath1 = scanpath_1[:,:2]
    if scanpath_2.shape[1] > 2:
        scanpath2 = scanpath_2[:,:2]

    rqaObj = RQA(scanpath_1, scanpath_2)
    
    determinism = rqaObj.determinism()
    laminarity = rqaObj.laminarity()
    corm = rqaObj.centerrecmass()
    crossrec = rqaObj.crossrec()

    return determinism, laminarity, corm, crossrec 



#########################################################################################

''' ScanMatch. Evaluating simirality between two fixation sequences with ScanMatch algorithm,
    proposed by Cristino, Mathot, Theeuwes and Gilchrist (2010). 
    Builds upon the GazeParser python package (https://pypi.org/project/GazeParser/)'''

from _scanMatch import ScanMatch


def scanMatch_metric(s1, s2, stimulus_width, stimulus_height, Xbin=14, Ybin=8, TempBin=50):

    if s1.shape[1] > 2 and s2.shape[1] > 2:
        tb = TempBin
        diff1 = (s1[:,3] - s1[:,2]) * 1000.
        s1 = np.hstack([s1[:,0].reshape(-1,1), s1[:,1].reshape(-1,1), diff1.reshape(-1,1)])
        diff2 = (s2[:,3] - s2[:,2]) * 1000.
        s2 = np.hstack([s2[:,0].reshape(-1,1), s2[:,1].reshape(-1,1), diff2.reshape(-1,1)])

    elif s1.shape[1] == 2 and s2.shape[1] == 2:
        tb = 0.0
    else:
        raise ValueError("Scanpaths should be arrays of shape [nfix,2] or [nfix,4]! Columns are assumed to be x,y fixations coordinates and, eventually, their start and end times (seconds)")
    
    matchObject = ScanMatch(Xres=stimulus_width, Yres=stimulus_height, Xbin=Xbin, Ybin=Ybin, TempBin=TempBin)
    seq1 = matchObject.fixationToSequence(s1).astype(int)
    seq2 = matchObject.fixationToSequence(s2).astype(int)

    (score, _, _) = matchObject.match(seq1, seq2)

    return score



#########################################################################################

''' MultiMatch. The MultiMatch method is a vector-based, multi-dimensional approach to compute scan path similarity. 
    It was originally proposed by Jarodzka, Holmqvist & Nyström (2010) and originally implemented as a Matlab toolbox by Dewhursts and colleagues (2012).
    Requires the multimatch-gaze python package (https://multimatch.readthedocs.io/en/latest/index.html)'''

import multimatch_gaze as mmg
import pandas as pd

def multiMatch_metric(s1, s2, stimulus_width, stimulus_height):

    assert s1.shape[1] > 2 and s2.shape[1] > 2, "Scanpaths should be arrays of shape [nfix,4] to compute MultiMatch! Columns are assumed to be x,y fixations coordinates and their their start and end times (seconds), resepectively"
    
    diff1 = (s1[:,3] - s1[:,2]) * 1000.
    ss1 = np.hstack([s1[:,0].reshape(-1,1), s1[:,1].reshape(-1,1), diff1.reshape(-1,1)])
    diff2 = (s2[:,3] - s2[:,2]) * 1000.
    ss2 = np.hstack([s2[:,0].reshape(-1,1), s2[:,1].reshape(-1,1), diff2.reshape(-1,1)])

    scan1_pd = pd.DataFrame({'start_x': ss1[:,0], 'start_y': ss1[:,1], 'duration': ss1[:,2]})
    scan2_pd = pd.DataFrame({'start_x': ss2[:,0], 'start_y': ss2[:,1], 'duration': ss2[:,2]})
    
    mm_scores = mmg.docomparison(scan1_pd.to_records(), scan2_pd.to_records(), screensize=[stimulus_width, stimulus_height])

    return mm_scores