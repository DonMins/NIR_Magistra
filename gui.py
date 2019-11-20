from tkinter import *
from tkinter import filedialog
from filter import Filter
import pandas as pd
import numpy as np
import sittings
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mne
from main import plotEEG
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



class mclass:
    def __init__(self,  window):
        self.window = window
        window.title("EEG analyzer")
        window.geometry('800x600')
        self.button2 = Button(window, bg="red", text=u"Кликни меня!", command=self.UploadAction)
        self.button2.pack()

    def UploadAction(self,event=None):
        filename = filedialog.askopenfilename()
        data = np.array(pd.read_csv(filename, sep=" ", header=None, skiprows=2))
        numSamples = 5 * sittings.FD
        data = data[0:numSamples]
        ticklocs = []
        fig = Figure()
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

        lines = LineCollection(segs, offsets=offsets, transOffset=None, colors="black")
        ax.add_collection(lines)

        ax.set_yticks(ticklocs)
        ax.set_yticklabels(
            ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'C4', 'T4', 'T5', 'P3', 'P4', 'T6', 'O1', 'O2'])

        ax.set_xlabel('Time (s)')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()


window= Tk()
start= mclass (window)
window.mainloop()