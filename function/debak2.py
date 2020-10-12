# coding: utf-8
#import STFTpycode as st
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

def fdicaflowtest( fname1 , fname2 ,fname3 ,fname4 ,itr):
    
    """--------------------------------------------------------------------------
    input  2 soundfile's name 
           itration
    
    output  estimation signal  
            estimation mel
            original signal 
            original mel
    
    --------------------------------------------------------------------------"""

    """--------------------------------------------------------------------------
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
    
    S1=S[:,0]
    S2=S[:,1]

    X1=X[:,0]
    X2=X[:,1]

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

    """--------------------------------------------------------------------------
    白色化
    --------------------------------------------------------------------------"""
    white_X = np.zeros([I,J,2],dtype="float")
    white_X = wh.whitening(stfted_S,2)

    print("whiteX(by X)= " + str(white_X.shape))
    print(np.argwhere(np.isnan(white_X)))
    print(white_X)
    
    #white_X = wh.whitening(stfted_S,2)
    #print("whiteX(by S)= " + str(white_X.shape))
    #print(np.argwhere(np.isnan(white_X)))
    #print(white_X)


    white_X1 = white_X[:,:,0] 
    white_X2 = white_X[:,:,1] 

    #print("white_X.shape =" +str(white_X.shape))

    """--------------------------------------------------------------------------
    fdica
    --------------------------------------------------------------------------"""
    kaisuu = int(itr)
    Y, estW = fdor.FDICA(white_X,kaisuu)

    print("Y.shape =" +str(Y.shape))    
    print("estW.shape =" +str(estW.shape))   

    """--------------------------------------------------------------------------
    backprojection & pps
    --------------------------------------------------------------------------"""
    #print(type(Y))
    #print(type(Y_r))
    #X = np.random.rand(513, 11)  # 4 x 4の配列の乱数
    estY,D= pbs.projection_back(Y, stfted_X[:,:,0])
    #Z=np.array(Z)

    pW =pps.perfectPermuSolver(estW,estY,stfted_S)

    Yp = np.zeros([2,J,I], dtype='float')
    Xwp = np.transpose(white_X, (2, 1, 0) )

    for i in range(I): # separation
        #Yp[:,:,i]= pW[:,:,i] * Xwp[:,:,i]
        Yp[:,:,i]= np.dot(pW[:,:,i],Xwp[:,:,i])

    Y = np.transpose(Yp, (2, 1, 0) )

    Z,D= pbs. projection_back(Y, stfted_X[:,:,0])
    
    
    est_y = stor.istft(Y, shiftSize, window_s, S.shape[0])
    est_z = stor.istft(Z, shiftSize, window_s, S.shape[0])
    
    S_out = stor.istft(stfted_S, shiftSize, window_s, S.shape[0])
    X_out = stor.istft(stfted_X, shiftSize, window_s, S.shape[0])
    print("All done new")
    a=1
    return est_z,Z,S_out,stfted_S,X_out,stfted_X,wavinfo

################################
# MAIN 
###############################
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
    wavf ='./dev1_male4_src_1_E2A_pos130_mic23.wav'
    iteration = 10000

    est_z,Z,s,stfted_S,x,stfted_X,wavinfo = fdicaflowtest( s1m1 ,s1m2 ,s2m1 ,s2m2 ,iteration)


  
    ################################
    # [outputs]
    # est_z    : estimation signal 
    # Z        : estimation spectrogram
    # s        : original signal
    # stfted_S : original spectrogram
    # x        : conv signal
    # stfted_X : conv spectrogram
    # wavinfo  : [ch1,width1,framerate,framenum] 
    #
    ###############################

    est_z =  np.array(est_z, dtype='int64')
    s     =  np.array(s, dtype='int64')
    x     =  np.array(x, dtype='int64')
 

    # 書き出し
    outf = './output/estimation/est1.wav'
    outd = struct.pack("h" * len(est_z[:,0]), *(est_z[:,0]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()

    # 書き出し
    outf = './output/estimation/est2.wav'
    outd = struct.pack("h" * len(est_z[:,1]), *(est_z[:,1]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()

        # 書き出し
    outf = './output/original/origin1.wav'
    outd = struct.pack("h" * len(s[:,0]), *(s[:,0]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()

    # 書き出し
    outf = './output/original/origin2.wav'
    outd = struct.pack("h" * len(s[:,1]), *(s[:,1]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()


    # 書き出し
    outf = './output/conv/conv1.wav'
    outd = struct.pack("h" * len(x[:,0]), *(x[:,0]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()

    # 書き出し

    outf = './output/conv/conv2.wav'
    outd = struct.pack("h" * len(x[:,1]), *(x[:,1]))
    ww = wave.open(outf, 'w')
    ww.setnchannels(wavinfo[0])
    ww.setsampwidth(wavinfo[1])
    ww.setframerate(wavinfo[2])
    ww.writeframes(outd)
    ww.close()


    print("debak2 is finished.")
