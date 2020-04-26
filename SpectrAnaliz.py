import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import entropy
import statistics as st

numSamples = 2000
time_step = 0.005
path = "EEG_Data\\Депрессия\\Norm\\" + str(4) + ".txt"
shiz = pd.read_csv(path, sep=" ", header=None, skiprows=2)
shiz = np.array(shiz[0:numSamples])

ps = np.abs(np.fft.ifft(shiz[:, 5]))
freqs = np.fft.fftfreq(numSamples, time_step)
idx = np.argsort(freqs)
idx = idx[int(numSamples/2):int(numSamples)]
newX = []
newY = []
for i in idx:
        newX.append(freqs[i])
        newY.append(ps[i])


waves = []

x=[i*time_step for i in range(2000)]
y=shiz[0:2000, 5]
plt.figure("Здоровые и шизофрения")
plt.rcParams.update({'font.size': 14})
plt.subplot(2, 1, 1)
plt.title("Фрагмент ЭЭГ сигнала")
plt.plot(x, y, color="black")
plt.xlabel(u'Время, с')
plt.ylabel(u'Амплитуда, мкВ')
plt.grid()

x2 = [i for i in np.arange(0,200,0.1)]
plt.subplot(2, 1, 2)
plt.title("Спектральная плотность мощности")
plt.plot(x2, ps*ps, color="black")
plt.grid()
plt.xlabel(u'Частота, Гц')
plt.ylabel(u'Плотноность мощности, кв.мкВ')
plt.subplots_adjust(wspace=0.1, hspace=0.5)

plt.show()