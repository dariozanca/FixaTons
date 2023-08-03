"""
.. Part of GazeParser package.
.. Copyright (C) 2012-2015 Hiroyuki Sogo.
.. Distributed under the terms of the GNU General Public License (GPL).

Evaluating simirality between two fixation sequences with ScanMatch algorithm,
proposed by Cristino, Mathot, Theeuwes and Gilchrist (2010).

Example
------------
Following script compares fixation sequence of two participants.::

    import GazeParser
    (data1, additionalData1) = GazeParser.load('participant1.db')
    (data2, additionalData2) = GazeParser.load('participant2.db')

    #create a ScanMatch object.
    matchObject = ScanMatch(Xres=720, Yres=720, Xbin=4, Ybin=4, offset=(152, 24), Threshold=1.5)

    #convert fixations to a sequence of symbols.
    sequence1 = sObj.fixationToSequence(data1[0].getFixationCenter())
    sequence2 = sObj.fixationToSequence(data2[0].getFixationCenter())

    #perform ScanMatch
    (score, align, f) = matchObject.match(sequence1, sequence2)

REFERENCE:
 Cristino, F., Mathot, S., Theeuwes, J., & Gilchrist, I. D. (2010).
 ScanMatch: a novel method for comparing fixation sequences.
 Behav Res Methods, 42(3), 692-700.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


class ScanMatch(object):
    """
    ScanMatch Object.
    """
    def __init__(self, **kw):
        """
        :param int Xres:
        :param int Yres:
        :param int Xbin:
        :param int Ybin:
        :param float Threshold:
        :param float GapValue:
        :param float TempBin:
        :param (int, int) Offset:
        """
        self.Xres = 1024
        self.Yres = 768
        self.Xbin = 8
        self.Ybin = 6
        self.Threshold = 3.5
        self.GapValue = 0.0
        self.TempBin = 0.0
        self.Offset = (0, 0)

        for k in kw.keys():
            if k == 'Xres':
                self.Xres = kw[k]
            elif k == 'Yres':
                self.Yres = kw[k]
            elif k == 'Xbin':
                self.Xbin = kw[k]
            elif k == 'Ybin':
                self.Ybin = kw[k]
            elif k == 'Threshold':
                self.Threshold = kw[k]
            elif k == 'GapValue':
                self.GapValue = kw[k]
            elif k == 'TempBin':
                self.TempBin = kw[k]
            elif k == 'Offset':
                self.Offset = kw[k]
            else:
                raise ValueError('Unknown parameter: %s.' % k)

        self.intv = numpy.vectorize(int)

        self.createSubMatrix()
        self.gridMask()

    def createSubMatrix(self, Threshold=None):
        if Threshold is not None:
            self.Threshold = Threshold
        mat = numpy.zeros((self.Xbin*self.Ybin, self.Xbin*self.Ybin))
        indI = 0
        indJ = 0
        for i in range(self.Ybin):
            for j in range(self.Xbin):
                for ii in range(self.Ybin):
                    for jj in range(self.Xbin):
                        mat[indI, indJ] = numpy.sqrt((j-jj)**2 + (i-ii)**2)
                        indI += 1
                indI = 0
                indJ += 1
        max_sub = numpy.max(mat)
        self.SubMatrix = numpy.abs(mat-max_sub) - (max_sub - self.Threshold)

    def gridMask(self):
        a = numpy.reshape(numpy.arange(self.Xbin*self.Ybin), (self.Ybin, self.Xbin))
        m = float(self.Xbin) / self.Xres
        n = float(self.Ybin) / self.Yres
        xi = numpy.int32(numpy.arange(0, self.Xbin, m))
        yi = numpy.int32(numpy.arange(0, self.Ybin, n))

        self.mask = numpy.zeros((self.Yres, self.Xres))
        for y in range(self.Yres):
            self.mask[y, :] = a[yi[y], xi]

    def fixationToSequence(self, data):
        d = data.copy()
        d[:, :2] -= self.Offset
        d[d < 0] = 0
        d[d[:, 0] >= self.Xres, 0] = self.Xres-1
        d[d[:, 1] >= self.Yres, 1] = self.Yres-1
        d = self.intv(d)

        seq_num = self.mask[d[:, 1], d[:, 0]]

        if self.TempBin != 0:
            fix_time = numpy.round(d[:, 2] / float(self.TempBin))
            tmp = []
            for f in range(d.shape[0]):
                tmp.extend([seq_num[f] for x in range(int(fix_time[f]))])
            seq_num = numpy.array(tmp)

        return seq_num

    def match(self, A, B):
        n = len(A)
        m = len(B)

        F = numpy.zeros((n+1, m+1))
        for i in range(n+1):
            F[i, 0] = self.GapValue*(i+1)
        for j in range(m+1):
            F[0, j] = self.GapValue*(j+1)

        for i in range(1, n+1):
            for j in range(1, m+1):
                match = F[i-1, j-1] + self.SubMatrix[A[i-1], B[j-1]]
                delete = F[i-1, j] + self.GapValue
                insert = F[i, j-1] + self.GapValue
                F[i, j] = max([match, insert, delete])

        AlignmentA = numpy.zeros(n+m)-1
        AlignmentB = numpy.zeros(n+m)-1
        i = n
        j = m
        step = 0

        while(i > 0 and j > 0):
            score = F[i, j]
            scoreDiag = F[i-1, j-1]
            # scoreUp = F[i, j-1]
            scoreLeft = F[i-1, j]

            if score == scoreDiag + self.SubMatrix[A[i-1], B[j-1]]:
                AlignmentA[step] = A[i-1]
                AlignmentB[step] = B[j-1]
                i -= 1
                j -= 1
            elif score == scoreLeft + self.GapValue:
                AlignmentA[step] = A[i-1]
                i -= 1
            else:
                AlignmentB[step] = B[j-1]
                j -= 1

            step += 1

        while(i > 0):
            AlignmentA[step] = A[i-1]
            i -= 1
            step += 1

        while(j > 0):
            AlignmentB[step] = B[j-1]
            j -= 1
            step += 1

        F = F.transpose()

        maxF = numpy.max(F)
        maxSub = numpy.max(self.SubMatrix)
        scale = maxSub * max((m, n))
        matchScore = maxF / scale

        align = numpy.vstack([AlignmentA[step-1::-1], AlignmentB[step-1::-1]]).transpose()

        return matchScore, align, F

    def maskFromArray(self, array):
        self.mask = array

    def subMatrixFromArray(self, array):
        self.SubMarix = array


def generateMaskFromArray(data, threshold, margeColor):
    dataArray = data.copy()
    uniqueData = numpy.unique(dataArray)
    for i in range(len(uniqueData)):
        index = numpy.where(dataArray == uniqueData[i])
        if len(index[0]) <= threshold:
            dataArray[index] = margeColor

    uniqueData2 = numpy.unique(dataArray)
    for i in range(len(uniqueData2)):
        index = numpy.where(dataArray == uniqueData2[i])
        dataArray[index] = i

    return dataArray, uniqueData2
