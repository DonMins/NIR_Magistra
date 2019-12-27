from filter import Filter
import pandas as pd
import numpy as np
import sittings
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
# import mne
from pylab import *
import pywt
from scipy.signal import hilbert, chirp
from scipy.interpolate import interp1d
import scaleogram as scg

for i in range(6,111):
    path = "EEG_Data\\tmp\\" + str(i+1) + ".txt"
    pathSave = "EEG_Data\\NormForAlcohol\\" + str(i+1) + ".txt"
    featureStringAll = np.zeros((256, 15))

    names = ['FP1', 'FP2', 'F7', 'F3', 'F4',
                 'F8', 'T3', 'C3', 'C4', 'T4',
                 'T5', 'P3', 'P4', 'T6', 'O1', 'O2']


    data = pd.read_csv(path, sep=" ", header=None,skiprows=4)

    T = []
    FP1 = []
    FP2 = []
    F7 = []
    F3 = []
    F4 = []
    T3 = []
    C3 = []
    C4 = []
    T4 = []
    T5 = []
    P3 = []
    P4 = []
    T6 = []
    O1 = []
    O2 = []


    for i in range(len(data[0])):
        if (data[1][i] =='FP1' and  data[0][i]!='#'):
            try:
                 T.append(i)
            except:
                pass
            try:
                FP1.append(data[3][i])
            except:
                pass

        elif (data[1][i] =='FP2'and  data[0][i]!='#'):
            try:
                FP2.append(data[3][i])
            except:
                pass

        elif (data[1][i] =='F7'and  data[0][i]!='#'):
            try:
                F7.append(data[3][i])
            except:
                pass

        elif (data[1][i] =='F3'and  data[0][i]!='#'):
            try:
                F3.append(data[3][i])
            except:
                pass

        elif (data[1][i] =='F4'and  data[0][i]!='#'):
            try:
                F4.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'T7'and  data[0][i]!='#'):
            try:
                T3.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'C3'and  data[0][i]!='#'):
            try:
                C3.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'C4'and  data[0][i]!='#'):
            try:
                C4.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'T8'and  data[0][i]!='#'):
            try:
                T4.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'P7'and  data[0][i]!='#'):
            try:
                T5.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'P3'and  data[0][i]!='#'):
            try:
                P3.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'P4'and  data[0][i]!='#'):
            try:
                P4.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'P8'and  data[0][i]!='#'):
            try:
                T6.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'O1'and  data[0][i]!='#'):
            try:
                O1.append(data[3][i])
            except:
                pass

        elif (data[1][i] == 'O2'and  data[0][i]!='#'):
            try:
                O2.append(data[3][i])
            except:
                pass

    featureStringAll[:,0] = T
    featureStringAll[:,1] = FP1
    featureStringAll[:,2] = FP2
    featureStringAll[:,3] = F7
    featureStringAll[:,4] = F3
    featureStringAll[:,5] = T3
    featureStringAll[:,6] = C3
    featureStringAll[:,7] = C4
    featureStringAll[:,8] = T4
    featureStringAll[:,9] = T5
    featureStringAll[:,10] = P3
    featureStringAll[:,11] = P4
    featureStringAll[:,12] = T6
    featureStringAll[:,13] = O1
    featureStringAll[:,14] = O2


    fileName = open(pathSave, 'w')
    np.savetxt(fileName, featureStringAll, fmt="%f")