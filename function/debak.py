
# coding: utf-8
import STFTpycode as st
import whitening as wh
import backpro as bp
import projection_sumino as pbs
import fdica
import fdica_or as fdor
import pps
import soundfile as sf
import stft_or as stor
import FDICA_FLOW as fdfl

import librosa.display
import pylab as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
import wave
import struct
from scipy import fromstring, int16

#import mir_eval

import struct
import math
import os
from scipy import fromstring, int16
import librosa
import librosa.display
import numpy as np
import time


fftSize=2048
shiftSize=1024
window_func='hamming'

s1m1 = "./dev1_female4_src_1_E2A_pos050_mic22.wav"
s1m2 = "./dev1_female4_src_1_E2A_pos050_mic23.wav"

s2m1 = "./dev1_male4_src_1_E2A_pos130_mic22.wav"
s2m2 = "./dev1_male4_src_1_E2A_pos130_mic23.wav"

time = 10
wavf ='./dev1_male4_src_1_E2A_pos130_mic23.wav'

fname1 = s1m1
fname2 = s1m2
fname3 = s2m1
fname4 = s2m2

wavf = fname1
wr = wave.open(wavf, 'r')

ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn1= wr.getnframes()

data = wr.readframes(wr.getnframes())
wr.close()
S1M1 = fromstring(data, dtype=int16)

wavf = fname2
wr = wave.open(wavf, 'r')
ch2 = wr.getnchannels()
width2 = wr.getsampwidth()
fr2 = wr.getframerate()
fn2= wr.getnframes()
data = wr.readframes(wr.getnframes())
wr.close()
S1M2 = fromstring(data, dtype=int16)
    
wavf = fname3
wr = wave.open(wavf, 'r')
ch3 = wr.getnchannels()
width3 = wr.getsampwidth()
fr3 = wr.getframerate()
fn3 = wr.getnframes()
data = wr.readframes(wr.getnframes())
wr.close()
S2M1 = fromstring(data, dtype=int16)
    
wavf = fname4
wr = wave.open(wavf, 'r')
ch4 = wr.getnchannels()
width4 = wr.getsampwidth()
fr4 = wr.getframerate()
fn4 = wr.getnframes()

data = wr.readframes(wr.getnframes())
wr.close()
S2M2 = fromstring(data, dtype=int16)
    
################################
# make conv signal   from 4-wavfiles
###############################
    
fn = min(fn1,fn2,fn3,fn4)
S_length = 160000
    
S1M1 = S1M1[0:S_length]
S1M2 = S1M2[0:S_length]
S2M1 = S2M1[0:S_length]
S2M2 = S2M2[0:S_length]
  
S = np.zeros([S_length,2])
X = np.zeros([S_length,2])
    
S[:,0] = S1M1[0:S_length]
S[:,1] = S2M1[0:S_length]
    
X[:,0] = S1M1 + S2M1
X[:,1] = S1M1 + S2M1
    

stfted_S, window_s = stor.stft(S, fftSize, shiftSize)
stfted_X, window_x = stor.stft(X, fftSize, shiftSize)

S_out = stor.istft(stfted_S, shiftSize, window_s, S.shape[0])
X_out = stor.istft(stfted_X, shiftSize, window_s, S.shape[0])

S_out =  np.array(S_out, dtype='int64')
X_out=  np.array(X_out, dtype='int64')






X_out = stor.istft(stfted_X, shiftSize, window_s, S.shape[0])
outf = './x1.wav'
outd = struct.pack("h" * len(X_out[:,0]), *(X_out[:,0]))
# 書き出し
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
ww.writeframes(outd)
ww.close()

outf = './x2.wav'

outd = struct.pack("h" * len(X_out[:,1]), *(X_out[:,1]))
# 書き出し
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
ww.writeframes(outd)
ww.close()

print(" saving is finished!")




