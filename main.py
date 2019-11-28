# from mne.preprocessing import ICA, create_ecg_epochs
# from mne.viz import plot_alignment, set_3d_view

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


def info(DATA_TIME, FD, N, NUMBER_CHANNELS, path):
    print("---------------------INFO About EEG Date--------------------")
    print("1. Name File ", path)
    print("2. Recording time: {0} c".format(DATA_TIME))
    print("3. Sampling frequency: {} Гц".format(FD))
    print("4. Сount in lines: {0} and channels: {1}".format(int(N), NUMBER_CHANNELS))
    print("------------------------------------------------------------")


def plotEEG(data, name, time, color='black'):
    numSamples = time * sittings.FD
    data = data[0:numSamples]
    ticklocs = []
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 10)

    numRows = len(data[0][0:]) - 1

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
        segs.append(np.hstack((t[:, np.newaxis], data[:, i + 1, np.newaxis])))
        ticklocs.append(i * dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None, colors=color, alpha=0.5)
    ax.add_collection(lines)

    ax.set_yticks(ticklocs)
    ax.set_yticklabels(
        ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'C4', 'T4', 'T5', 'P3', 'P4', 'T6', 'O1', 'O2'])

    ax.set_xlabel('Time (s)')
    plt.tight_layout()


def EegChart(data, NUMBER_CHANNELS, N_DATA, FREQUENCY):
    mas = np.zeros((NUMBER_CHANNELS, N_DATA))
    for i in range(1, NUMBER_CHANNELS + 1):
        allMax = 0
        mmax = abs(max(data[i][0:N_DATA]))
        mmin = abs(min(data[i][0:N_DATA]))
        allMax = max(mmax, mmin)

        for j in range(N_DATA):
            mas[i - 1][j] = (10 * (data[i][j]) / allMax)  # для нормального отображение графиков сделаем нормировку

    data = np.array([mas[0], mas[1], mas[2], mas[3], mas[4], mas[5], mas[6], mas[7],
                     mas[8], mas[9], mas[10], mas[11], mas[12], mas[13], mas[14], mas[15]])
    info = mne.create_info(
        ch_names=['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg'],
        sfreq=FREQUENCY
    )
    raw = mne.io.RawArray(data, info)
    scalings = {'eeg': NUMBER_CHANNELS}

    calings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    raw.plot(n_channels=16, scalings=scalings, title='Auto-scaled Data from arrays',
             show=True, block=True)


def average(data, time):
    x = np.linspace(time[0], time[1], len(data))
    yMax = []
    newxMax = []
    arrayY = 0

    yMin = []
    newxMin = []
    for i in range(len(data) - 2):
        j = i + 1
        m = j + 1

        i1 = (data[i])
        i2 = (data[j])
        i3 = (data[m])

        if (((i2) >= (i1)) and ((i2) >= (i3))):
            newxMax.append(x[j])
            yMax.append(i2)

        if (((i2) <= (i1)) and ((i2) <= (i3))):
            newxMin.append(x[j])
            yMin.append(i2)

    return newxMax, yMax, newxMin, yMin, arrayY


def approximation(data, canal, time):
    start = time[0] * sittings.FD
    end = time[1] * sittings.FD

    s = data[start:end, canal]
    x = np.linspace(time[0], time[1], end - start)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    f = average(s, [0, end - start])
    u_x = f[0]
    u_y = f[1]

    l_x = f[2]
    l_y = f[3]

    # Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)
    l_p = interp1d(l_x, l_y, kind='cubic', bounds_error=False, fill_value=0.0)

    # Evaluate each model over the domain of (s)
    for k in range(0, len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)

    plt.plot(x, s, alpha=0.4, color='black', label='Исходный сигнал')
    plt.plot(x, q_u, '#0000FF', label='Верняя огибающая')
    plt.plot(x, q_l, '#008000', label='Нижняя огибающая')
    plt.plot(x, (q_u + q_l) / 2, '--', color='red', label='Среднее')
    plt.grid()
    plt.legend()
    # plt.show()

    # ----- Тоже самое, но на отдельных графиках ------ #
    numSamples1 = time[0] * sittings.FD
    numSamples2 = time[1] * sittings.FD
    numSamples = numSamples2 - numSamples1

    x = np.linspace(time[0], time[1], numSamples)
    y = data[numSamples1:numSamples2, canal]
    plt.plot(x, y, color='black', alpha=0.5)

    plt.subplot(4, 1, 1)
    plt.plot(x, y, color="black")
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Фрагмент ЭЭГ сигнала (F7)")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")

    plt.subplot(4, 1, 2)
    f = average(data[numSamples1:numSamples2, canal], time)
    #
    plt.plot(x, y, color="black", alpha=0.5)
    plt.plot(f[0], (f[1]), 'o', color='blue', markersize=3)
    plt.plot(f[2], (f[3]), 'o', color='red', markersize=3)
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Экстремумы сигнала (F7)")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    plt.subplot(4, 1, 3)
    f = average(data[numSamples1:numSamples2, canal], time)
    #
    plt.plot(x, y, color="black", alpha=0.5)
    plt.plot(x, q_u, color='blue', markersize=1)
    plt.plot(x, q_l, color='red', markersize=1)
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Две огибающие,построенные по максимумам и минимумам")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    plt.subplot(4, 1, 4)
    plt.plot(x, y, color='black', alpha=0.5)
    plt.plot(x, (q_u + q_l) / 2, '--', color='red')
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Тренд сигнала")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    h1 = s - ((q_u + q_l) / 2)
    plt.plot(x, s)
    plt.grid()
    print(f[4])
    plt.show()


def isModa(data):
    numberOfExtremes = 0
    numberOfIntersections = 0
    for i in range(len(data) - 2):
        j = i + 1
        m = j + 1

        i1 = (data[i])
        i2 = (data[j])
        i3 = (data[m])

        if (((i2) >= (i1)) and ((i2) >= (i3))) or (((i2) <= (i1)) and ((i2) <= (i3))):
            numberOfExtremes += 1

        if (i1 < 0 and i2 >= 0) or (i1 >= 0 and i2 < 0):
            numberOfIntersections += 1
    if (numberOfExtremes - numberOfIntersections <= 1):
        return True
    return False


def isCorrectAvg(avg):
    if (-0.5 < np.mean(avg) < 0.5):
        return True
    return False


def getModa(s, time):  # возвращает моду из набора данных
    start = time[0] * sittings.FD
    end = time[1] * sittings.FD
    m = 0
    h1 = []
    while (m < 10):
        q_u = np.zeros(s.shape)
        q_l = np.zeros(s.shape)

        f = average(s, [0, end - start])
        u_x = f[0]
        u_y = f[1]

        l_x = f[2]
        l_y = f[3]

        # Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

        # u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0)
        # l_p = interp1d(l_x, l_y, kind='cubic', bounds_error=False, fill_value=0)

        u_p = CubicSpline(u_x, u_y, bc_type='clamped')
        l_p = CubicSpline(l_x, l_y, bc_type='clamped')

        # Evaluate each model over the domain of (s)
        for k in range(0, len(s)):
            q_u[k] = u_p(k)
            q_l[k] = l_p(k)

        avg = (q_u + q_l) / 2
        h1 = s - avg  # 1 приближение к моде
        h1[len(h1) - 1] = s[len(s) - 1]
        h1[0] = s[0]
        if (isModa(h1) == True and isCorrectAvg(avg) == True):
            break
        else:
            s = h1
        m += 1
    return h1


def SpectrFurie(data):
    time_step = 0.005
    ps = np.abs(np.fft.ifft(data))
    freqs = np.fft.fftfreq(len(data), time_step)
    idx = np.argsort(freqs)
    idx = idx[int(len(data) / 2):int(len(data))]
    newX = []
    newY = []
    for i in idx:
        # if freqs[i] >= 0 and freqs[i] <= 100:
             newX.append(freqs[i])
             newY.append(ps[i])
    return newX, newY




def getEmpiricalFashion(data, canal, time):
    start = time[0] * sittings.FD
    end = time[1] * sittings.FD
    s = data[start:end, canal]
    x = np.linspace(time[0], time[1], end - start)

    st = data[start:end, canal]

    plt.subplot(2, 2, 1)
    plt.grid()
    path = "EEG_Data\\MAN\\20-33\\" + str("1") + ".txt"
    DATA_TIME = 40
    FD = 200
    N = DATA_TIME / (1 / FD)
    datas = np.array(pd.read_csv(path, sep=" ", header=None, skiprows=2))
    maxel = max(datas[:,3])
    datas[:,3] =   datas[:,3]/maxel


    plt.plot(x, st, label="Исходный сигнал", color='black')
    plt.plot(x, datas[start:end, canal], label="Исходный сигнал", color='green')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()

    plt.subplot(2, 2, 2)
    spect = SpectrFurie(st)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Спектр исходного сигнала")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")
    wavelet = 'morl'

    ax = scg.cws(x, st, scales=range(1,int(len(x)/2)), wavelet=wavelet,yaxis='frequency',
                 figsize=(14, 3), cmap="jet",  ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()

    plt.subplot(2, 2, 1)
    h1 = getModa(s, time)
    plt.grid()
    plt.plot(x, st, label="Исходный сигнал", color='black', alpha = 0.5)
    plt.plot(x, h1, label="Мода 1", color='red')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()

    plt.subplot(2, 2, 2)
    spect = SpectrFurie(h1)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Cпектр 1 моды")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")

    ax = scg.cws(x, h1, scales=range(1, int(len(x) / 2)), wavelet=wavelet, yaxis='frequency',
                 figsize=(14, 3), cmap="jet", ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()
    r1 = s - h1

    plt.subplot(2, 2, 1)
    h2 = getModa(r1, time)
    plt.plot(x, st, label="Исходный сигнал", color='black', alpha = 0.5)
    plt.plot(x, h2, label="Мода 2", color='red')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    spect = SpectrFurie(h2)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Спектр 2 моды")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")

    ax = scg.cws(x, h2, scales=range(1, int(len(x) / 2)), wavelet=wavelet, yaxis='frequency',
                 figsize=(14, 3), cmap="jet", ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()
    r2 = r1 - h2

    plt.subplot(2, 2, 1)
    h3 = getModa(r2, time)
    plt.plot(x, st, label="Исходный сигнал", color='black',alpha = 0.5)
    plt.plot(x, h3, label="Мода 3", color='red')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 2)
    spect = SpectrFurie(h3)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Cпектр 3 моды")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")

    ax = scg.cws(x, h3, scales=range(1, int(len(x) / 2)), wavelet=wavelet, yaxis='frequency',
                 figsize=(14, 3), cmap="jet", ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()
    r3 = r2 - h3

    plt.subplot(2, 2, 1)
    h4 = getModa(r3, time)
    plt.plot(x, st, label="Исходный сигнал", color='black',alpha = 0.5)
    plt.plot(x, h4, label="Мода 4", color='red')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    spect = SpectrFurie(h4)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Спектр 4 моды")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")

    ax = scg.cws(x, h4, scales=range(1, int(len(x) / 2)), wavelet=wavelet, yaxis='frequency',
                 figsize=(14, 3), cmap="jet", ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()
    r4 = r3 - h4

    plt.subplot(2, 2, 1)
    h5 = getModa(r4, time)
    #plt.plot(x, st, label="Исходный сигнал", color='black',alpha = 0.5)
    plt.plot(x, h5, label="Мода 5 ", color='red')
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 2)
    spect = SpectrFurie(h5)
    plt.plot(spect[0], spect[1], color='black')
    plt.title("Cпектр 5 моды")
    plt.grid()
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда (МкВ)")

    ax = scg.cws(x, h5, scales=range(1,int(len(x)/2)), wavelet=wavelet,yaxis='frequency',
                 figsize=(14, 3), cmap="jet",  ylabel='Период (год)', xlabel="Время [Год]",
                 title='Вейвлет-преобразование временного ряда\n(спектр мощности)')

    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(x, st, label="Сигнал с наложением артефакта", alpha=0.5, color='black')
    plt.subplot(3, 1, 2)
    plt.plot(x, (h1 +h2+ h3),  label="Отфильрованный сигнал", color = 'red')
    plt.xlabel("Время(с)")
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.ylabel("Амплитуда (МкВ)")
    path = "EEG_Data\\MAN\\20-33\\" + str("1") + ".txt"
    DATA_TIME = 40
    FD = 200
    N = DATA_TIME / (1 / FD)
    data = np.array(pd.read_csv(path, sep=" ", header=None, skiprows=2))
    maxel = max(data[:, 3])
    data[:, 3] = data[:, 3] / maxel
    plt.subplot(3, 1, 3)
    plt.plot(x, data[800:1600, 3], label="Сигнал без наложения артефакта" ,color = 'blue')
    plt.subplot(3, 1, 1)
    plt.plot(x, st, label="Сигнал с наложением артефакта", alpha=0.5, color='black')
    plt.plot(x, data[800:1600, 3], label="Сигнал без наложения артефакта" ,color = 'blue')

    #
    # minY = np.percentile(st, q=[10, 80])[0]
    # maxY = np.percentile(st, q=[10, 80])[1]
    #
    # for i in range(len(st)):
    #     if st[i] >= maxY:
    #         st[i] = maxY
    #     if st[i] <= minY:
    #         st[i] = minY
    #     continue
    # plt.plot(x, st, color='blue', label="Отфильрованный сигнал c помощью робастого преобразовния")
    plt.legend()
    # plt.grid()
    plt.show()


# ------- робастое преобразование------------
def percentile(data, canal, time):
    numSamples = time * sittings.FD
    x = [i for i in range(numSamples)]
    y = data[0:numSamples, canal]
    plt.plot(x, y, color='black', alpha=0.5)

    minY = np.percentile(y, q=[10, 90])[0]
    maxY = np.percentile(y, q=[10, 90])[1]

    for i in range(len(y)):
        if y[i] >= maxY:
            y[i] = maxY
        if y[i] <= minY:
            y[i] = minY
        continue

    plt.plot(x, y)


# -------------------------------------

def plotSingleCanal(data, canal, time):
    numSamples1 = time[0] * sittings.FD
    numSamples2 = time[1] * sittings.FD
    numSamples = numSamples2 - numSamples1

    x = np.linspace(time[0], time[1], numSamples)
    y = data[numSamples1:numSamples2, canal]
    plt.plot(x, y, color='black', alpha=0.5)

    plt.subplot(4, 1, 1)
    plt.plot(x, y, color="black")
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Фрагмент ЭЭГ сигнала (F7)")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")

    plt.subplot(4, 1, 2)
    f = average(data[numSamples1:numSamples2, canal], time)
    #
    plt.plot(x, y, color="black", alpha=0.5)
    plt.plot(f[0], (f[1]), 'o', color='blue', markersize=3)
    plt.plot(f[2], (f[3]), 'o', color='red', markersize=3)
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Экстремумы сигнала (F7)")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    plt.subplot(4, 1, 3)
    f = average(data[numSamples1:numSamples2, canal], time)
    #
    plt.plot(x, y, color="black", alpha=0.5)
    plt.plot(f[0], (f[1]), marker='o', color='blue', markersize=3)
    plt.plot(f[2], (f[3]), marker='o', color='red', markersize=3)
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Две огибающие,построенные по максимумам и минимумам")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    plt.subplot(4, 1, 4)
    t = min(len(f[0]), len(f[2]))
    print(len(f[0]), "  ", len(f[2]))
    trend = []

    for i in range(t):
        trend.append((f[1][i] + f[3][i]) / 2)

    newTrend = []
    newX = []

    print(len(trend))

    for i in range(len(trend) - 1):
        newTrend.append(trend[i])
        newTrend.append((trend[i] + trend[i + 1]) / 2)
        newX.append(f[2][i])
        newX.append((f[2][i] + f[2][i + 1]) / 2)

    print(len(newTrend))

    plt.plot(x, y, color='black', alpha=0.5)
    plt.plot(newX, newTrend, color='red')
    plt.axvline(x=6, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.axvline(x=7.5, ymin=0, ymax=400, linewidth=1, linestyle='dashed', color='green')
    plt.title("Тренд сигнала")
    plt.xlabel("Время(с)")
    plt.ylabel("Амплитуда (МкВ)")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # minY= np.percentile(y, q=[10, 90])[0]
    # maxY= np.percentile(y, q=[10, 90])[1]
    #
    # for i in range(len(y)):
    #     if y[i]>=maxY:
    #         y[i] = maxY
    #     if y[i]<=minY:
    #         y[i] = minY
    #     continue
    #
    # plt.plot(x,y)

    # f = average(data[0:numSamples,canal])
    #
    # plt.plot(f[0], (hilbert(f[1])),  color='blue', markersize=3)
    # plt.plot(f[2],(hilbert(f[3])),  color = 'red', markersize=3)
    # plt.xlabel("Времся (с)")
    # plt.show()
    #
    # t = min(len(f[0]),len(f[2]))
    # print(len(f[0]),"  ", len(f[2]))
    # trend = []
    #
    # for i in range(t):
    #     trend.append((f[1][i]+f[3][i])/2)
    #
    # newTrend = []
    # newX=[]
    #
    # print(len(trend))
    #
    # for i in range(len(trend)-1):
    #     newTrend.append(trend[i])
    #     newTrend.append((trend[i]+trend[i+1])/2)
    #     newX.append(f[0][i])
    #     newX.append((f[0][i]+f[0][i+1])/2)
    #
    # print(len(newTrend))
    #
    # plt.plot(x,y,color ='black' ,alpha=0.5)
    # plt.plot(newX, newTrend, color='red')
    plt.show()


if __name__ == "__main__":
    # ---------------Данные инициализации-----------------------------------

    path = "EEG_Data\\MAN\\20-33\\" + str("1") + ".txt"
    DATA_TIME = 40
    FD = 200
    N = DATA_TIME / (1 / FD)
    data = np.array(pd.read_csv(path, sep=" ", header=None, skiprows=2))
    NUMBER_CHANNELS = len(data[0][0:]) - 1

    info(DATA_TIME=DATA_TIME, FD=FD, path=path, N=N, NUMBER_CHANNELS=NUMBER_CHANNELS)

    # ------------------------------------------------------------------------

    # -------------------Фильтрация-------------------------------------------
    data[1200:1250, :] = data[1200:1250, :] + 100
    data[1250:1300, :] = data[1250:1300, :] + 200
    data[1300:1350, :] = data[1300:1350, :] + 280
    data[1350:1400, :] = data[1350:1400, :] + 320

    data[1400:1450, :] = data[1400:1450, :] + 300
    data[1450:1500, :] = data[1450:1500, :] + 130
    data[1550:1600, :] = data[1550:1600, :] + 50

    # plotEEG(data,"ЭЭГ сигнал с артефактом (движение глаз)",10)
    # for i in range(1,17):
    #     minY = np.percentile(data[:, i], q=[10, 90])[0]
    #     maxY = np.percentile(data[:, i], q=[10, 90])[1]
    #
    #     for j in range(len(data[:, i])):
    #         if data[j, i] >= maxY:
    #             data[j, i] = maxY
    #         if data[j, i] <= minY:
    #             data[j, i] = minY
    #         continue
    #
    # #
    # plotEEG(data, "Удаление артефактов робастым преобразвоанием", 10)
    #
    # plt.grid()
    # filter_data = Filter().Fourier_filter(data=data,ranges= 'alpha')
    # plotEEG(filter_data,"FILTER_alpha_EEG",10)
    # plt.grid()
    # raw = EegChart(pd.read_csv(path, sep=" ", header=None, skiprows=2),NUMBER_CHANNELS,int(N),FD)
    # # raw2 = EegChart(pd.read_csv(path, sep=" ", header=None, skiprows=2),NUMBER_CHANNELS,int(N),FD)
    # plt.show()
    # plotSingleCanal(data, 3, [4,8])

    #approximation(data,3, [4,8])

    from random import gauss
    # from random import seed
    #
    #
    # # seed random number generator
    # seed(1)
    # # create white noise series
    # # series = [gauss(0.0, 1000.0) for i in range(len(data[:,3] ))]
    # # data[:,3] =   data[:,3] + series
    #
    # Fs = len(data[:, 3])  # sampling rate
    # Ts = 1.0 / Fs  # sampling interval
    # t = np.arange(0, 1, Ts)  # time vector
    #
    # ff = 20  # frequency of the signal
    # y = np.sin(2 * np.pi * ff * t)
    #
    # maxel = max(data[:, 3])
    # data[:, 3] = data[:, 3] / maxel + y

    getEmpiricalFashion(data, 3, [4, 8])

