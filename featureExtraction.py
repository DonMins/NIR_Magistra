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

def corrEnvelope(x, y):
    """Метод возвращает корреляцию Пирсона между двумя каналами
    Входные данные: список точек 1 и 2 каналов
    """
    amplitude_envelopeX = (hilbert(x))
    amplitude_envelopeY = (hilbert(y))
    envelopeArrayX = np.abs(amplitude_envelopeX)
    envelopeArrayY = np.abs(amplitude_envelopeY)
    corrPirson = np.corrcoef(envelopeArrayX,envelopeArrayY)[0,1]
    if (math.isnan(corrPirson)): return 0.0
    return corrPirson

def calculateCorrelation(data):
    """Метод считает межполушарную синхронность"""
    featureString = []
    featureString.append(corrEnvelope(data['Fp1'], data['Fp2']))
    featureString.append(corrEnvelope(data['F7'], data['F8']))
    featureString.append(corrEnvelope(data['F3'], data['F4']))
    featureString.append(corrEnvelope(data['T3'], data['T4']))
    featureString.append (corrEnvelope(data['C3'], data['C4']))
    featureString.append(corrEnvelope(data['T5'], data['T6']))
    featureString.append(corrEnvelope(data['P3'], data['P4']))
    featureString.append(corrEnvelope(data['O1'], data['O2']))
    return featureString


def getFeatures(data, featureStringAll,i,className = None):
    featureString = featureStringAll
    # -----------Получили признаки из альфа диапазона---------------#
    alphaData = Filter().Fourier_filter(data=np.array(data), ranges='alpha',className =className)
    alphaData = pd.DataFrame(data=alphaData[:, :], columns=names)
    featureStringAlpha = calculateCorrelation(alphaData)

    # -----------Получили признаки из дельта диапазона---------------#
    deltaData = Filter().Fourier_filter(data=np.array(data), ranges='delta',className =className)
    deltaData = pd.DataFrame(data=deltaData[:, :], columns=names)
    featureStringDelta = calculateCorrelation(deltaData)

    # -----------Получили признаки из тета диапазона---------------#
    thetaData = Filter().Fourier_filter(data=np.array(data), ranges='theta',className =className)
    thetaData = pd.DataFrame(data=thetaData[:, :], columns=names)
    featureStringTheta = calculateCorrelation(thetaData)

    # -----------Получили признаки из бета1 диапазона---------------#
    beta1Data = Filter().Fourier_filter(data=np.array(data), ranges='beta1',className =className)
    beta1Data = pd.DataFrame(data=beta1Data[:, :], columns=names)
    featureStringBeta1 = calculateCorrelation(beta1Data)

    # -----------Получили признаки из бета2 диапазона---------------#
    beta2Data = Filter().Fourier_filter(data=np.array(data), ranges='beta2',className =className)
    beta2Data = pd.DataFrame(data=beta2Data[:, :], columns=names)
    featureStringBeta2 = calculateCorrelation(beta2Data)
    if className!=None:
        featureString[i - 1,:] = featureStringAlpha + featureStringBeta1 + featureStringBeta2 + featureStringDelta + featureStringTheta + className
    else:
        featureString[i - 1,:] = featureStringAlpha + featureStringBeta1 + featureStringBeta2 + featureStringDelta + featureStringTheta

    return featureString

if __name__ == "__main__":
    #featureStringAll = np.zeros((30, 40)) # агрессивные
    #featureStringAll = np.zeros((103, 40)) # женщины

    names = ['T[sec]', 'Fp1', 'Fp2', 'F7', 'F3', 'F4',
             'F8', 'T3', 'C3', 'C4', 'T4',
             'T5', 'P3', 'P4', 'T6', 'O1', 'O2']
    pathSave = "EEG_Features\\" + str(1) + ".txt"

    # featureStringAll = np.zeros((76, 41))  # мужчины
    # for i in range(1,77):
    #     path = "EEG_Data\\MAN\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None, skiprows=2,names = names )
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [0]
    #     featureStringAll = getFeatures(data, featureStringAll,i , clas)
    #
    # fileName = open(pathSave, 'w')
    # np.savetxt(fileName, featureStringAll, fmt="%f")
    #
    # featureStringAll = np.zeros((103, 41)) # женщины
    # for i in range(1, 104):
    #     path = "EEG_Data\\FEMALE\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names)
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [0]
    #     featureStringAll = getFeatures(data, featureStringAll,i ,clas)
    #
    # fileName = open(pathSave, 'a')
    # np.savetxt(fileName, featureStringAll, fmt="%f")
    #
    # featureStringAll = np.zeros((60, 41)) # NORM
    # names2 = ['T[sec]',  'F7',  'F3',  'F4',  'F8',  'T3',  'C3',  'Cz',  'C4',  'T4',  'T5',  'P3',  'Pz',  'P4',  'T6',  'O1',  'O2']
    # for i in range(1, 61):
    #     path = "EEG_Data\\Norm\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names2)
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [0]
    #     featureStringAll = getFeatures(data, featureStringAll,i,clas)
    #
    # fileName = open(pathSave, 'a')
    # np.savetxt(fileName, featureStringAll, fmt="%f")
    #
    # featureStringAll = np.zeros((50, 41)) # depress
    # names2 = ['T[sec]',  'F7',  'F3',  'F4',  'F8',  'T3',  'C3',  'Cz',  'C4',  'T4',  'T5',  'P3',  'Pz',  'P4',  'T6',  'O1',  'O2']
    # for i in range(1, 51):
    #     path = "EEG_Data\\depress\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None, skiprows=2, names=names2)
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [1]
    #     featureStringAll = getFeatures(data, featureStringAll,i,clas)
    #
    # fileName = open(pathSave, 'a')
    # np.savetxt(fileName, featureStringAll, fmt="%f")

    # featureStringAll = np.zeros((3000, 41))  # Алкоголики
    # for i in range(1,3001):
    #     path = "EEG_Features\\Алко\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None,names = names )
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [1]
    #     featureStringAll = getFeatures(data, featureStringAll,i , clas)
    #
    # fileName = open(pathSave, 'w')
    # np.savetxt(fileName, featureStringAll, fmt = '%.5f')
    # fileName.close()
    #
    # featureStringAll = np.zeros((3000, 41)) # Не алкоголики
    # for i in range(1, 3001):
    #     path = "EEG_Features\\Не алко\\" + str(i) + ".txt"
    #     data = pd.read_csv(path, sep=" ", header=None,  names=names)
    #
    #     data['className'] = [0 for i in range(len(data['T[sec]']))]
    #     clas = [0]
    #     featureStringAll = getFeatures(data, featureStringAll,i ,clas)
    #
    # fileName = open(pathSave, 'a')
    # np.savetxt(fileName, featureStringAll, fmt = '%.5f')
    # fileName.close()

    featureStringAll = np.zeros((6, 40))  # Test
    names2 = ['T[sec]', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    for i in range(7000, 7006):
        path = "EEG_Features\\Алко\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None,  names=names2)

        featureStringAll = getFeatures(data, featureStringAll,i - 7000)

    fileName = open("EEG_Features\\test.txt", 'w')
    np.savetxt(fileName, featureStringAll, fmt = '%.5f')
    fileName.close()

    featureStringAll = np.zeros((6, 40))  # Test
    names2 = ['T[sec]', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    for i in range(3001, 3007):
        path = "EEG_Features\\Не алко\\" + str(i) + ".txt"
        data = pd.read_csv(path, sep=" ", header=None, names=names2)

        featureStringAll = getFeatures(data, featureStringAll, i - 3001)

    fileName = open("EEG_Features\\test.txt", 'a')
    np.savetxt(fileName, featureStringAll, fmt='%.5f')
    fileName.close()








