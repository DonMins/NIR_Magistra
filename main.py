from filter import Filter
import pandas as pd
import numpy as np
import sittings
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def info(DATA_TIME,FD, N,NUMBER_CANALS, path):
    print("---------------------INFO About EEG Date--------------------")
    print("1. Name File ", path)
    print("2. Recording time: {0} c".format(DATA_TIME))
    print("3. Sampling frequency: {} Гц".format(FD))
    print("4. Сount in lines: {0} and channels: {1}".format(int(N),NUMBER_CANALS))
    print("------------------------------------------------------------")

def plotEEG(data,name,time):
    numSamples = time * sittings.FD
    data = data[0:numSamples]
    ticklocs = []
    fig = plt.figure(name)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(0, 10)

    numRows=len(data[0][0:])-1

    t = 10.0 * np.arange(numSamples) / numSamples
    ax.set_xticks(np.arange(10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows - 1) * dr + dmax
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:, np.newaxis], data[:, i+1, np.newaxis])))
        ticklocs.append(i * dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None,colors="black")
    ax.add_collection(lines)

    ax.set_yticks(ticklocs)
    ax.set_yticklabels(['Fp1','Fp2','F7', 'F3', 'F4', 'F8', 'T3', 'C3','C4', 'T4', 'T5', 'P3', 'P4', 'T6', 'O1', 'O2'])

    ax.set_xlabel('Time (s)')

    plt.tight_layout()


if __name__ == "__main__":

    #---------------Данные инициализации-----------------------------------

    path = "EEG_Data\\MAN\\20-33\\" + str(1) + ".txt"
    DATA_TIME = 40
    FD = 200
    N = DATA_TIME/(1/FD)
    data = np.array(pd.read_csv(path, sep=" ", header=None, skiprows=2))
    NUMBER_CANALS = len(data[0][0:])-1

    info(DATA_TIME = DATA_TIME, FD=FD,path=path,N=N,NUMBER_CANALS=NUMBER_CANALS)

    #------------------------------------------------------------------------

    #-------------------Фильтрация-------------------------------------------
    plotEEG(data,"EEG",5)
    filter_data = Filter().Fourier_filter(data=data,ranges= 'alpha')
    plotEEG(filter_data,"FILTER_alpha_EEG",5)
    plt.show()