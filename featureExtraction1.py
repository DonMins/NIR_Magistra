

# Здесь создается матрица признаков на основе Статистического анализ сигнала (папка Признаки - (1)).


import matplotlib.pyplot as plt
import pandas as pd
import os
#import mne
import os.path as op
import numpy as np
import math
from collections import deque
#from mne.minimum_norm import read_inverse_operator, compute_source_psd
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, chirp
from scipy import signal
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import statistics as st
from scipy.stats import chi2
from sklearn.svm import SVC
from  filter import  Filter
from main import plotEEG

def getFeatures(data, featureStringAll,i,countCanal,className = None):
    featureString = featureStringAll
    for j in range(1,countCanal + 1):
        featureString[i - 1, j - 1] = np.var(np.array(data)[:,j])

    if className!=None:
        featureString[i - 1, countCanal] =  className
    return  featureString



if __name__ == "__main__":

    names = ['T[sec]','F7', 'F3', 'F4','F8', 'T3', 'C3', 'Cz', 'C4','T5', 'P3', 'Pz','P4', 'T6', 'O1', 'O2']
    pathSave = "Признаки\\1\\" + str("allVall") + ".txt"

    featureStringAll = np.zeros((61, 16))
    for i in range(1,62):
        path = "EEG_Data\\Депрессия\\depress\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, skiprows=2,names = names )

        # data['className'] = [1 for i in range(len(data['T[sec]']))]  # 1 - больной , 0 - здоровый
        clas = np.array(3)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'w')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()

    featureStringAll = np.zeros((60, 16))
    for i in range(1, 61):
        path = "EEG_Data\\Депрессия\\Norm\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(0)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()



    featureStringAll = np.zeros((30, 16))
    for i in range(1, 31):
        path = "EEG_Data\\prisoners_aggressive\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(1)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()


    featureStringAll = np.zeros((7033, 16))
    for i in range(1, 7034):
        path = "EEG_Data\\Алкоголики\\Алко\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(2)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()

    featureStringAll = np.zeros((3921, 16))
    for i in range(1, 3922):
        path = "EEG_Data\\Алкоголики\\Не алко\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(0)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()


    featureStringAll = np.zeros((76, 16))
    for i in range(1, 77):
        path = "EEG_Data\\Просто_здоровые_люди\\MAN\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(0)
        featureStringAll = getFeatures(data, featureStringAll,i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()


    featureStringAll = np.zeros((103, 16))
    for i in range(1, 104):
        path = "EEG_Data\\Просто_здоровые_люди\\FEMALE\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names)

        # data['className'] = [0 for i in range(len(data['T[sec]']))]
        clas = np.array(0)
        featureStringAll = getFeatures(data, featureStringAll, i, 15, clas)

    fileName = open(pathSave, 'a')
    np.savetxt(fileName, featureStringAll, fmt="%f")
    fileName.close()

