__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"




import os
import torch as th
import scipy.io
import numpy as np
import scipy.fftpack as ft
# import matplotlib.pyplot as plt
# import tkinter



def save_checkpoint(state, path, filename):

    th.save(state, os.path.join(path, filename))
    
    
    
    
    
    
    
def timeToFrequency(SampleRate,FrequencyRange,AcousticMeasurment):
    Frame = 1 / SampleRate
    L = len(AcousticMeasurment)

    Time = Frame*L;
    time = np.linspace(0.0, Time , L)

    #print(time[1],max(time))

    Y = ft.fft(AcousticMeasurment)
    Freq = np.linspace(0.0, 1.0 / (2.0 * Frame), L // 2)
    
    P = 2.0 / L * np.abs(Y[0:L // 2])
    
   
    FrequencyRange_native = [1e5, 8e5]

    f1_native = Freq >= FrequencyRange_native[0]
    f2_native = Freq <= FrequencyRange_native[1]
    f_native  = f1_native*f2_native

    
    
    f1 = Freq >= FrequencyRange[0]
    f2 = Freq <= FrequencyRange[1]
    f  = f1*f2
    P = P[f] #*20
    Freq = Freq[f]


   # print(6*np.max(P))

    # interpolate to bigger input
    Freq_new = np.linspace(Freq[0],Freq[-1],np.sum(f_native))
    P_new = np.interp(Freq_new,Freq,P)


    return P_new, Freq_new
    
    
    
    
    
def frequencyFilter(SampleRate,FrequencyRange,AcousticMeasurment):
    Frame = 1 / SampleRate
    L = len(AcousticMeasurment)

    Time = Frame*L;
    time = np.linspace(0.0, Time , L)

    #print(time[1],max(time))

    Y = ft.fft(AcousticMeasurment)
    Freq = np.linspace(0.0, 1.0 / (2.0 * Frame), L // 2)
    mFreq = -Freq[::-1]
 
    f1 = Freq < FrequencyRange[0]
    f2 = Freq > FrequencyRange[1]
    mf1 = mFreq > -FrequencyRange[0]
    mf2 = mFreq < -FrequencyRange[1]


    f1 = np.concatenate((f1, mf1)).reshape(-1)
    f2 = np.concatenate((f2, mf2)).reshape(-1)

    Y[f1] = 0
    Y[f2] = 0
   
    
    AcousticMeasurment_Limit = ft.ifft(Y).real

#    print(10*np.max(AcousticMeasurment_Limit))



    return AcousticMeasurment_Limit
