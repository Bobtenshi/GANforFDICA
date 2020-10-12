
import os
from sklearn import preprocessing
import cloudpickle
import sys
sys.path.append('/home/s-yamaji/.pyenv/versions/3.6.8/lib/python3.6/site-packages')
sys.path.append('/home/s-yamaji/.local/lib/python3.6/site-packages')
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
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import csv

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

            for freq in range(20,row-20):
                #initial setting------------------
                input_data = np.zeros([50,4], dtype='float32')#入力ベクトルのサイズ
                wide = 50#アクティベーションの長さ
                start_point = random.randint(0,colum-wide-1)#取得開始時間フレーム
                target_ch = random.randint(0,nch)#ターゲットch決め
                #make input vector-------------------
                ch_order = list(range(nch))
                random.shuffle(ch_order)

                for index,ch in enumerate(ch_order):
                    input_data[:,index] = Z_power[freq,start_point:start_point + wide,ch]#正規化いる？
                    if ch == target_ch:
                        input_data[:,-1] = Z_power[freq + random.randint(0,15),start_point:start_point + wide,target_ch]
                        self.label.append(torch.eye(nch)[index])
                submit = np.reshape(input_data, (-1))
                self.data.append(submit)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]
        #if self.transform:
            #out_data = self.transform(out_data)
        return out_data, out_label


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        hidden1 = 256
        hidden2 = 64
        self.fc1 = nn.Linear( 50*4, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, hidden1)
        self.fc4 = nn.Linear(hidden1, hidden2)
        self.fc5 = nn.Linear(hidden2, hidden2)
        self.fc6 = nn.Linear(hidden2, hidden2)
        self.fc7 = nn.Linear(hidden2, hidden2)
        self.fc8 = nn.Linear(hidden2, hidden2)
        self.fc9 = nn.Linear(hidden2, hidden2)
        self.fc10 = nn.Linear(hidden2, 3)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = F.relu(self.fc3(h))
        h = self.dropout(h)
        h = F.relu(self.fc4(h))
        h = self.dropout(h)
        h = F.relu(self.fc5(h))
        h = self.dropout(h)
        h = F.relu(self.fc6(h))
        #h = F.relu(self.fc7(h))
        #h = F.relu(self.fc8(h))
        #h = F.relu(self.fc9(h))
        #h = self.dropout(h)
        out = self.fc10(h)
        #F.log_softmax(self.fc5(h), dim = 0)

        return F.log_softmax(out, dim = 1)


if __name__ == "__main__":

    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'

    model = Mymodel()
    model = model.to(device)

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    loss_log = [] # 学習状況のプロット用
    acc_log = [] # 学習状況のプロット用
    train_loss = []
    train_accu = []
    i = 0
    epoch = 5
    batchsize  = 128 # 1つのミニバッチのデータの数
    t1 = time.time()

    with open('./Output/dataset/dataset_est_target_1ch.pickle', 'rb') as f:
        data_set = pickle.load(f)

    train_dataset, valid_dataset = torch.utils.data.random_split(  # データセットの分割
    data_set,   # 分割するデータセット
    [int(len(data_set)/2), len(data_set)-int(len(data_set)/2)])  # 分割数

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,  # データセットの指定
        batch_size=batchsize,  # ミニバッチの指定
        shuffle=True,  # シャッフルするかどうかの指定
        num_workers=0)  # コアの数

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    print('train_dataset = ', len(train_dataset))
    print('valid_dataset = ', len(valid_dataset))
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    train_loss, train_acc, val_loss, val_acc,t = 0, 0, 0, 0,0
    accu_array =np.zeros([60,2])
    while val_acc<0.95:
    #while t<50:
        t+=1
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        # ======== train_mode ======
        model.train() #学習モード
        a = train_dataset[0]
        x = a[0].shape
        for xt, yt in train_loader: # 1ミニバッチずつ計算
            x,y = yt.shape
            #if y!=52:
            #    print("error")
            #xt = np.log(xt)
            xt, yt = Variable(xt), Variable(yt)#微分可能な型
            xt = xt.cuda()
            yt = yt.cuda()

            optimizer.zero_grad()
            y_model = model(xt)

            loss = criterion(y_model, yt)
            loss.backward()    #バックプロパゲーション
            optimizer.step()   # 重み更新
            #train_loss.append(loss.data.item())

            pred_label = y_model.data.max(1)[1] #予測結果
            accu_label = yt.data.max(1)[1] #予測結果
            train_acc = torch.sum(pred_label==accu_label).cpu().numpy()/float(x)
            #maxpred = max(pred_label)
            #minpred = min(pred_label)
        print(t, "  train_accu =>"+str(train_acc),"  train_Loss =>"+str(loss.item()))
        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc)
        accu_array[t,0]=train_acc

        # ======== valid_mode ======
        model.eval() #学習モード
        for xt, yt in valid_loader: # 1ミニバッチずつ計算

            xt, yt = Variable(xt), Variable(yt)#微分可能な型
            xt = xt.cuda()
            yt = yt.cuda()
            x,y = yt.shape
            with torch.no_grad():
                output = model(xt)
                loss = criterion(output, yt)
                pred_label = output.data.max(1)[1] #予測結果
                accu_label = yt.data.max(1)[1] #予測結果
                #a = yt.view(-1).int()
                val_acc = torch.sum(pred_label==accu_label).cpu().numpy()/float(x)
        print(t, "  val_accu =>"+str(val_acc),"  val_Loss =>"+str(loss.item()))
        val_loss_list.append(loss.item())
        val_acc_list.append(val_acc)
        accu_array[t,1]=val_acc
        if val_acc>=0.998:
            break

    t2 = time.time()
    print(f"経過時間：{t2-t1}")

    #plt.savefig("./fdica/output/fig/dnn1t_Loss.jpg")
    #if not os.path.exists('./dnn_all_label/output/fig/'):
    #    os.mkdir('./dnn_all_label/output/fig/')
    #plt.plot(train_loss_list)
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.yscale('log')
    #plt.savefig('/../../dnn_all_label/output/fig/dnn_same_train_loss.jpg')
    #plt.clf()
    with open('./make_dnn_models/output/trained_model/model_ref'+str(ref)+'.pkl', 'wb') as f:
        cloudpickle.dump(model, f)
    #with open('./make_dnn_models/output/trained_model/accu.csv', 'w') as f:
    #    writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
    #    writer.writerows(accu_array)  # 2次元配列も書き込める