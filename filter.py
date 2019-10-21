import pandas as pd
from scipy.fftpack import fft, ifft
import numpy as np
import sittings

class Filter:
    def __init__(self):
        pass

    def Fourier_filter(self,data,ranges,FD=None):
        if FD==None: FD = sittings.FD
        ranges = sittings.Frequency_Ranges[ranges]
        lower_freq = ranges[0]
        high_freq = ranges[1]
        N = int(sittings.N)
        number_channels = len(data[0][0:])-1
        print(number_channels)
        answer = np.array(np.zeros((N,number_channels+1)))
        answer[0:N,0] = data[0:N,0]

        x = np.fft.fftfreq(N, 1./FD)

        for k in range (number_channels):
            fourier_array = (np.fft.fft(data[0:N,k+1]))

            for i in range(int(N)):
                if x[i] >= high_freq:
                    fourier_array[i] = 0
                if x[i] <= lower_freq:
                    fourier_array[i] = 0

            result = np.fft.ifft(fourier_array).real
            answer[0:N, k+1] = result[0:N]
        return answer


