#!/usr/bin/env python
# coding: utf-8
from function import whitening as wh
from function import backpro as bp
from function import projection_sumino as pbs
from function import fdica
from function import fdica_or as fdor
from function import pps
from function import stft_or as stor
import soundfile as sf
import librosa.display
import pylab as plt
import numpy as np
from scipy import signal
import wave
import struct
from scipy import fromstring, int16
def perfectPermuSolver(Wr,Y,S):
    #import torch
    #from torch.autograd import Variable

    I,J,M = Y.shape
    pW = np.zeros([M,M,I], dtype='complex')

    #pW=torch.from_numpy(pW)
    #Wr=torch.from_numpy(Wr)


    if M == 2:
        for i in range(I):
            dist1 = sum(np.power(abs(Y[i,:,0] - S[i,:,0]),2)) + sum(np.power(abs(Y[i,:,1] - S[i,:,1]),2))
            dist2 = sum(np.power(abs(Y[i,:,1] - S[i,:,0]),2)) + sum(np.power(abs(Y[i,:,0] - S[i,:,1]),2))

            if dist1 <= dist2:
                pW[:,:,i] = Wr[:,:,i]
            else:
                pW[0,:,i] = Wr[1,:,i]
                pW[1,:,i] = Wr[0,:,i]

    return pW

