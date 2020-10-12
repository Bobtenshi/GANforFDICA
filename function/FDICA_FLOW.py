
# coding: utf-8

# In[2]:


import STFTpycode as st
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
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import signal
#from sklearn.decomposition import PCA
#import torch
#from torch.autograd import Variable
import wave
import struct
from scipy import fromstring, int16

def fdicaflow( fname1 , fname2 ,fname3 ,fname4 ,itr):
    
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
    wavf = fname1
    wr = wave.open(wavf, 'r')

    ch1 = wr.getnchannels()
    width1 = wr.getsampwidth()
    fr1 = wr.getframerate()
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
    
    #print(S.shape)


    A = np.array([[0.8, 0.2], 
                  [0.5, 0.7],
                 ])  # Mixing matrix

    #X = np.dot(S, A.T)  # Generate observations

    S1=S[:,0]
    S2=S[:,1]

    X1=X[:,0]
    X2=X[:,1]

    #print("X_type = "+str(type(X)))
    #print("X_size = "+str(X.shape))

    """--------------------------------------------------------------------------
    STFT
    --------------------------------------------------------------------------"""

    stfted_S, window_s = stor.stft(S, fftSize, shiftSize)
    stfted_X, window_x = stor.stft(X, fftSize, shiftSize)

    I,J,M = stfted_S.shape

    
    #print("stfted_S.shape =" +str(stfted_S.shape))
    """--------------------------------------------------------------------------
    白色化
    --------------------------------------------------------------------------"""
    white_X = np.zeros([I,J,2],dtype="float")
    white_X = wh.whitening(stfted_X,2)

    white_X1 = white_X[:,:,0] 
    white_X2 = white_X[:,:,1] 
    #print("white_X.shape =" +str(white_X.shape))

    """--------------------------------------------------------------------------
    fdica
    --------------------------------------------------------------------------"""
    kaisuu = int(itr)
    
    #Y1,Y2, estW, hist_loss = fdica.FDICA(white_X1,white_X2,kaisuu )
    Y, estW = fdor.FDICA(white_X,kaisuu)
    #gyou = int(Y1.shape[0])
    #retu = int(Y1.shape[1])

    #Y = np.zeros([gyou, retu,2], dtype="float")
    #for i in range(gyou):
    #    Y[i,:,0] = Y1[i,:]
    #    Y[i,:,1] = Y2[i,:]

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
    pW = pW.detach().numpy()

    Yp = np.zeros([2,J,I], dtype='float')
    Xwp = np.transpose(white_X, (2, 1, 0) )

    for i in range(I): # separation
        #Yp[:,:,i]= pW[:,:,i] * Xwp[:,:,i]
        Yp[:,:,i]= np.dot(pW[:,:,i],Xwp[:,:,i])

    Y = np.transpose(Yp, (2, 1, 0) )

    
    
    Z,D= pbs. projection_back(Y, stfted_X[:,:,0])
    
    
    est_y = stor.istft(Y, shiftSize, window_s, S.shape[0])
    #est_z = stor.istft(Z, shiftSize, window_s, S.shape[0])
    
    S_out = stor.istft(stfted_S, shiftSize, window_s, S.shape[0])
    X_out = stor.istft(stfted_X, shiftSize, window_s, S.shape[0])
    print("All done new")
    
    a=1
    #return est_z,Z,S_out,stfted_S,X_out,stfted_X
    return est_y,a,S_out,stfted_S,X_out,stfted_X
    #Z=np.array(Z)
    #print(type(Z))
    #print(Z.shape)
    #print(Z.ndim)


# In[ ]:




