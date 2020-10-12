import librosa.display
import pylab as plt
import numpy as np
from scipy import signal
import wave
import struct
from scipy import fromstring, int16
import mir_eval
import random
import pickle
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import cloudpickle
from function import stft_or as stft
from scipy import hamming
from scipy.io import wavfile

def ps_use_dnn(Y):
    I,J,M = Y.shape
    datatens = np.zeros([I,J,M], dtype='complex128')
    complex_datatens = np.zeros([I,J,M], dtype='complex128')


    datatens[:,:,0] = Y[:,:,0]
    datatens[:,:,1] = Y[:,:,1]

    complex_datatens[:,:,0]= datatens[:,:,0]
    complex_datatens[:,:,1]= datatens[:,:,1]



    with open('./output/trained_model/dnn_same_model.pkl', 'rb') as f:
        model = cloudpickle.load(f)

    print(model)
    model = model.eval()

    if torch.cuda.is_available():  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    #good = 0
    #errorindex=[]
    predlabel =[]
    frec_bin = I-1
    for p in range(frec_bin):
        testdata =[]
        for i in np.arange(0,datatens.shape[1]-6, 5):
            #p = 61

            testdatatens = np.zeros([6,4], dtype='float32')
            testdatatens[:,0] =  np.reshape(datatens[p  ,i:i+6,0],(-1))
            testdatatens[:,1] =  np.reshape(datatens[p+1,i:i+6,0],(-1))
            testdatatens[:,2] =  np.reshape(datatens[p  ,i:i+6,1],(-1))
            testdatatens[:,3] =  np.reshape(datatens[p+1,i:i+6,1],(-1))

            testdata.append(np.reshape(testdatatens[:,:],(-1)))

        test_loader = torch.utils.data.DataLoader(testdata, batch_size=1)

        j,k = 0,0
        for xt in test_loader: # 1ミニバッチずつ計算
            xt= Variable(xt)#微分可能な型
            xt = xt.cuda()
            with torch.no_grad():
                output = model(xt)
                pred_label = output.data.max(1)[1] #予測結果
                #print(pred_label)
                if pred_label == 0:
                    j= j+1
                else:
                    k=k+1
        if j>=k:
            predlabel.append(0)
        else:
            predlabel.append(1)

        print("###############################")
        print("###############################")
        print("  p =>"+str(p))
        print("  zero =>"+str(j),"  one =>"+str(k))
        #datatens[p:p+2,-1,:]
        #print("  ラベル=>"+str(datatens[p:p+2,-1,0]))

    #print(" 正解ビン数=>"+str(good))
    #print(" 不正解ビン数=>"+str(frec_bin-good))
    #print(errorindex)


    #sortlabel = predlabel
    predlabel = np.array(predlabel)
    sortlabel = predlabel.copy()
    for index, label in enumerate(predlabel):
        sortlabel[index] = sum(predlabel[:index+1])

    #sortlabel = np.array(sortlabel)
    sortlabel = sortlabel % 2

    sorttens = np.zeros([I,J,M], dtype='complex128')
    sorttens[0,:,0]=complex_datatens[0,:,0]
    sorttens[0,:,1]=complex_datatens[0,:,1]

    for index, label in enumerate(sortlabel):
        if label==1:#入れ替え
            sorttens[index+1,:,0]=complex_datatens[index+1,:,1]
            sorttens[index+1,:,1]=complex_datatens[index+1,:,0]
        elif label==0:#入れ替えなし
            sorttens[index+1,:,0]=complex_datatens[index+1,:,0]
            sorttens[index+1,:,1]=complex_datatens[index+1,:,1]


    stft.spectrogram(np.array(sorttens[:,:,0],dtype=float),output_path='./output/fig/sort_dnn_same/sort_spec1')
    stft.spectrogram(np.array(sorttens[:,:,1],dtype=float), output_path='./output/fig/sort_dnn_same/sort_spec2')
    return sorttens
