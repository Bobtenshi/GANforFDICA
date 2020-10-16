import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
from tqdm import tqdm
import os
import pickle
import statistics
import random
import glob
from Toolbox.stft_or import stft,spectrogram,istft,power
from PIL import Image

class Load_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        file_num = glob.glob('./Output/fdiced_Z_spec/2_ch/*')

        for i in tqdm(range(2**5)):
            #load specs
            with open(random.choice(file_num), 'rb') as f:
                specs = pickle.load(f)#4097,121,3
            #mat化
            input_data = np.zeros([2,64,32],dtype='float32')#[ch,heigh weight]

            #img_matrix[0,:,:] = np.power(specs[:,:,0],2)
            #img_matrix[1,:,:] = np.power(specs[:,:,1],2)
            input_data[0,:,:] = power(specs[128:128+64,64:64+32,0])
            input_data[1,:,:] = power(specs[128:128+64,64:64+32,1])
            #正規化?

            input_data[0,:,:] = input_data[0,:,:]  / np.linalg.norm(input_data[0,:,:] , ord=2)
            input_data[1,:,:] = input_data[1,:,:]  / np.linalg.norm(input_data[1,:,:] , ord=2)

            # to cuda
            data = torch.from_numpy(np.reshape(input_data, (-1))).clone().to("cuda:0")
            self.data.append(data)
        print(str(len(self.data))+'_data made')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        #if self.transform:
            #out_data = self.transform(out_data)
        return out_data


#################################################
#################################################
#################################################

class Generator(nn.Module):
    # input : 1x96x96 の Y [-1, 1] 本当は[0, 1]
    # output : 2x96x96 の CrCb [-1, 1] 本当は[-0.5, 0.5]
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(64*32*2,64),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,64),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,64))
        self.enc2 = nn.Sequential(
            nn.Linear(64+64*32*2,64*64*2),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64*64*2,64*32*2))

    def forward(self, x):
        #encode
        out = self.enc1(x)#(-1,16,2048,32)
        out = self.enc2(torch.cat([out, x], dim=1))#(-1,16,2048,32)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(4096,128),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,64),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,32),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32,8),nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8,1),nn.Sigmoid())
    def forward(self, x):
        #encode
        out = self.enc1(x)#(-1,16,2048,32)
        return out

#################################################
#################################################
#################################################

def normalize(values, lower, upper):
    return (values - lower) / (upper - lower)
def denormalize(values, lower=0, upper=1):
    return values * (upper - lower) + lower
def to_grayscale(values, lower, upper):
    normalized = normalize(values, lower, upper)
    denormalized = np.clip(denormalize(normalized, 0, 255), 0, 255)
    return denormalized.astype(np.float32)

def ips2fdica(real_ips_cuda):
    real_ips = real_ips_cuda.to('cpu').detach().numpy().copy()
    real_fdica = np.zeros_like(real_ips)#(batchsize,ch,freq,time-frame)
    A = real_ips.shape[2]
    for batch_n in range(real_ips.shape[0]):#batch
        for freq in range(real_ips.shape[2]):#freq
            if freq < real_ips.shape[2]/2:
                real_fdica[batch_n,0,freq,:] = real_ips[batch_n,0,freq,:].copy()
                real_fdica[batch_n,1,freq,:] = real_ips[batch_n,1,freq,:].copy()
            else:
                real_fdica[batch_n,0,freq,:] = real_ips[batch_n,1,freq,:].copy()
                real_fdica[batch_n,1,freq,:] = real_ips[batch_n,0,freq,:].copy()
    return torch.from_numpy(real_fdica).clone().to("cuda:0")

def train():
    # initial setting
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    model_G, model_D = Generator(), Discriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)
    params_G = torch.optim.Adam(model_G.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    # ロスを計算するためのラベル変数 (PatchGAN)
    batch_size = 1
    ones = torch.ones(batch_size, 1).to(device)
    zeros = torch.zeros(batch_size, 1).to(device)
    # 損失関数
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.MSELoss()
    # エラー推移
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []


    #Load data
    dataset = Load_Dataset()
    data_loader = torch.utils.data.DataLoader(
    dataset=dataset,  # データセットの指定
    batch_size=batch_size,  # ミニバッチの指定
    shuffle=True,  # シャッフルするかどうかの指定
    num_workers=0)  # コアの数


    #Train
    for i in range(200):
        log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
        #for real_ips, _ in tqdm(dataset):
        for real_ips in tqdm(data_loader):
            batch_len = len(real_ips)
            real_ips = real_ips.to(device)
            #real_fdica = ips2fdica(real_ips)#シャッフル
            real_fdica = real_ips.clone()

            #確認用
            # Gの訓練
            # 偽のカラー画像を作成
            fake_ips = model_G(real_fdica.view(batch_size, -1))#batch,ch,fight=4097,wideth=121?
            #fake_ips_fdica = torch.cat([real_fdica[:,:1,:,:], fake_ips], dim=1)
            #fake_ips_fdica = fake_ips#無変換

            # 偽画像を一時保存
            fake_ips_fdica_tensor = fake_ips.detach()

            # 偽画像を本物と騙せるようにロスを計算
            out = model_D(fake_ips)#batchsizwe,2,fight=4097,wideth=128?
            loss_G_bce = bce_loss(out, ones[:batch_len])#batch,1,128,2
            #print(loss_G_bce)
            #a = torch.cat([fake_ips[:,0,:,:],fake_ips[:,1,:,:]],dim=2)

            e1 = mae_loss(fake_ips,real_fdica.view(batch_size, -1))#誤差１

            e2 = mae_loss(fake_ips,real_fdica.view(batch_size, -1))#誤差１

            if e1<e2:
                loss_G_mae = e1
            else:
                loss_G_mae = e2
            #loss_G_mae = 50 * mae_loss(fake_ips, real_fdica[:, :,:,:]) + 50 * mae_loss(fake_ips_fdica, real_ips)
            #loss_G_mae = 30 * mae_loss(e1, e2) + 70 * mae_loss(fake_ips, real_ips)
            loss_G_sum = loss_G_bce + loss_G_mae

            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G_sum.backward()
            params_G.step()

            # Discriminatoの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = model_D(real_ips.view(batch_size, -1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像を偽と識別できるようにロスを計算
            fake_out = model_D(fake_ips_fdica_tensor)
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()
            writer.add_scalar("Loss/Loss_G", loss_G_sum,i)
            writer.add_scalar("Loss/Loss_D", loss_D,i)
            writer.close()

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print(f"log_loss_G_sum = {result['log_loss_G_sum'][-1]} " +
              f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
              f"log_loss_D = {result['log_loss_D'][-1]}")

        # 生成画像を保存
        # モデルの保存

        if i % 5 == 0 or i == 199:
            if not os.path.exists(f"./Output/train_result/models/epoch_{i:03}/"):
                os.mkdir(f"./Output/train_result/models/epoch_{i:03}/")
            torch.save(model_G.state_dict(), f"./Output/train_result/models/epoch_{i:03}/gen_{i:03}.pytorch")
            torch.save(model_D.state_dict(), f"./Output/train_result/models/epoch_{i:03}/dis_{i:03}.pytorch")
            #torchvision.utils.save_image(real_ips[0,0,],f"./Output/train_result/models/real_epoch_{i:03}.png")


            #save_grayscale(real_ips[0,0,:,:],i)
            spec_img_real = real_ips.view(batch_size, 64,-1)
            spec_img_fake = fake_ips.view(batch_size, 64,-1)

            spectrogram(spec_img_real[0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real_{i:03}")
            spectrogram(spec_img_fake[0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake_{i:03}")
            #spectrogram(real_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real2_{i:03}")
            #spectrogram(fake_ips[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake1_{i:03}")
            #spectrogram(fake_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake2_{i:03}")
            #spectrogram(real_fdica[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fdica1_{i:03}")
            #spectrogram(real_fdica[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fdica2_{i:03}")

    # ログの保存
    with open(f"./Output/train_result/models/epoch_{i:03}/logs.pickle", "wb") as fp:
        pickle.dump(result, fp)
    writer.flush()


def save_grayscale(spec,i):
    spec_np = spec.detach().cpu().numpy()
    lower,upper = np.amin(spec_np),np.amax(spec_np)
    normalized = normalize(spec_np, lower, upper)
    denormalized = np.clip(denormalize(normalized, 0, 255), 0, 255)
    img = Image.fromarray(denormalized.astype(np.uint8))
    img.save(f"./Output/train_result/models/real_epoch_{i:03}.png")

    #torchvision.utils.save_image(real_ips[0,0,],f"./Output/train_result/models/real_epoch_{i:03}.png")
    #return denormalized.astype(np.float32)


if __name__ == "__main__":
    train()