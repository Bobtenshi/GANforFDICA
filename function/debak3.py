# coding: utf-8
#import STFTpycode as st
import whitening as wh
import backpro as bp
import projection_sumino as pbs
import fdica
import fdica_or as fdor
import pps
import soundfile as sf
import stft_or as stor

import librosa.display
import pylab as plt
import numpy as np
from scipy import signal
import wave
import struct

from scipy import fromstring, int16


if __name__ == "__main__":
    fftSize=2048
    shiftSize=1024
    window_func='hamming'

    """tell file name"""

    s1m1 = "./imput/dev1_female4_src_1_E2A_pos050_mic22.wav"
    s1m2 = "./imput/dev1_female4_src_1_E2A_pos050_mic23.wav"
    s2m1 = "./imput/dev1_male4_src_1_E2A_pos130_mic22.wav"
    s2m2 = "./imput/dev1_male4_src_1_E2A_pos130_mic23.wav"
    time = 10

    fname1 = s1m1
    fname2 = s1m2
    fname3 = s2m1
    fname4 = s2m2
    iteration = 10000

    """--------------------------------------------------------------------------
    input  2 soundfile's name 
           itration
    
    output  estimation signal  
            estimation mel
            original signal 
            original mel
    
    --------------------------------------------------------------------------"""

    """---------------------
    -----------------------------------------------------
    初期値設定
    --------------------------------------------------------------------------"""
    fftSize=2048
    shiftSize=1024
    window_func='hamming'
    
    """--------------------------------------------------------------------------
    混合信号作成
    --------------------------------------------------------------------------"""
    ################################
    # wavfile loding 
    ################################
    """read wav1"""
    wavf = fname1
    wr = wave.open(wavf, 'r')
    ch1 = wr.getnchannels()
    width1 = wr.getsampwidth()
    fr1 = wr.getframerate()
    fn1= wr.getnframes()

    data = wr.readframes(wr.getnframes())
    wr.close()
    S1M1 = fromstring(data, dtype=int16)

    wavinfo = [ch1,width1,fr1,fn1]

    """read wav2"""
    wavf = fname2
    wr = wave.open(wavf, 'r')
    ch2 = wr.getnchannels()
    width2 = wr.getsampwidth()
    fr2 = wr.getframerate()
    fn2= wr.getnframes()
    data = wr.readframes(wr.getnframes())
    wr.close()
    S1M2 = fromstring(data, dtype=int16)

    """read wav3"""
    wavf = fname3
    wr = wave.open(wavf, 'r')
    ch3 = wr.getnchannels()
    width3 = wr.getsampwidth()
    fr3 = wr.getframerate()
    fn3 = wr.getnframes()
    data = wr.readframes(wr.getnframes())
    wr.close()
    S2M1 = fromstring(data, dtype=int16)
    
    """read wav4"""
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
    """--------------------------------------------------------------------------
    STFT
    --------------------------------------------------------------------------"""

    stfted_S, window_s = stor.stft(S, fftSize, shiftSize)
    print(np.argwhere(np.isnan(stfted_S)))
    
    stfted_X, window_x = stor.stft(X, fftSize, shiftSize)
    print(np.argwhere(np.isnan(stfted_X)))
    
    
    power_X = abs(stfted_X)**2
    sumg = np.sum(power_X, axis=0)
    sumr = np.sum(power_X, axis=1)
    
    count_g_0 = 2*len(sumg) - np.count_nonzero(sumg)
    count_r_0 = 2*len(sumr) - np.count_nonzero(sumr)
    
    print("stfted_X.shape =" + str(stfted_X.shape))
    print("0 X num 列 = " + str(count_g_0))
    print("0 X num 行 = " + str(count_r_0))
    
    print("len(sumr) = " + str(len(sumr)))
    print("len(sumg) = " + str(len(sumg)))

    power_S = abs(stfted_S)**2
    
    sumg = np.sum(power_S, axis=0)
    sumr = np.sum(power_S, axis=1)
    
    count_g_0 = 2*len(sumg) - np.count_nonzero(sumg)
    count_r_0 = 2*len(sumr) - np.count_nonzero(sumr)
    
    print("stfted_S.shape =" + str(stfted_S.shape))
    print("0 S num  列= " + str(count_g_0))
    print("0 S num  行 = " + str(count_r_0))

    I,J,M = stfted_S.shape
    #print("stfted_S.shape =" +str(stfted_S.shape))
    
    print("debak3 is finished.")
