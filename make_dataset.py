import soundfile as sf
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        #self.data_num = data_num
        self.data = []
        self.label = []

        file_list = glob.glob('./Output/fdiced_Z_spec/*')
        #loading spec------------------
        for file_name in file_list:
            print(file_name)
            with open( file_name, 'rb') as f:
                Z_comp = pickle.load(f)
                Z_power = stft.power(Z_comp)
            row,colum,nch = Z_comp.shape

            for freq in range(20,row-20,5):#ほぼ全周波数走査
                #initial setting------------------
                input_data = np.zeros([50,4], dtype='float32')#入力ベクトルのサイズ
                wide = 50#アクティベーションの長さ
                start_point = random.randint(0,colum-wide-1)#取得開始時間フレーム
                target_ch = random.randint(0,nch-1)#ターゲットch決め
                #make input vector-------------------
                ch_order = list(range(nch))
                random.shuffle(ch_order)#chの順番をランダムに

                for index,ch in enumerate(ch_order):
                    print("index:{}  ch:{}".format(index,ch))
                    input_data[:,index] = Z_power[freq,start_point:start_point + wide,ch]#正規化いる？
                    if ch == target_ch:
                        input_data[:,-1] = Z_power[freq + random.randint(0,15),start_point:start_point + wide,target_ch]
                        self.label.append(torch.eye(nch)[index])
                submit = np.reshape(input_data, (-1))
                self.data.append(submit)
        #print(self.data[10])
        #print(self.label[10])

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]
        #if self.transform:
            #out_data = self.transform(out_data)
        return out_data, out_label


def dataset_dnn_same_fdica():
    data_set = Dataset()
    #data_set = Dataset_dnn_same_fdica(512)
    with open('./Output/dataset/dataset_est_target_1ch.pickle', 'wb') as f:
        pickle.dump(data_set,f)
    f.close

if __name__ == "__main__":
    dataset_dnn_same_fdica()


