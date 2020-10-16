import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
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

        for i in tqdm(range(2**7)):
        #while len(self.data)<=2**10-1:#2**n個のデータセット
            #print(len(self.data))
            #load specs
            with open(random.choice(file_num), 'rb') as f:
                specs = pickle.load(f)#4097,121,3
            #mat化
            img_matrix = np.zeros([2,specs.shape[0],specs.shape[1]],dtype='float32')#[ch,heigh weight]

            #img_matrix[0,:,:] = np.power(specs[:,:,0],2)
            #img_matrix[1,:,:] = np.power(specs[:,:,1],2)
            img_matrix[0,:,:] = np.abs(specs[:,:,0])
            img_matrix[1,:,:] = np.abs(specs[:,:,1])
            #正規化?
            img_matrix

            #gray_ips_matrix = to_grayscale(log_img_matrix, np.amin(log_img_matrix), np.amax(log_img_matrix))
            gray_ips_matrix = img_matrix/np.max(img_matrix)
            #gray_ips_matrix = gray_ips_matrix /255
            # to cuda
            data = torch.from_numpy(gray_ips_matrix).clone().to("cuda:0")
            self.data.append(data)
        print(str(len(self.data))+'_data made')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        #if self.transform:
            #out_data = self.transform(out_data)
        return out_data

class Generator(nn.Module):
    # input : 1x96x96 の Y [-1, 1] 本当は[0, 1]
    # output : 2x96x96 の CrCb [-1, 1] 本当は[-0.5, 0.5]
    def __init__(self):
        super().__init__()

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1, 128, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(1+1, 1, kernel_size=3, stride=1, padding=1,output_padding=0),
            nn.Sigmoid())


    def forward(self, x):
        #encode
        out = self.dec1(x)#(-1,16,2048,32)
        out = self.dec2(out)#(-1,64,1024,16)
        out = self.dec3(out)#(-1,128,512,8)
        out = self.dec4(out)#(-1,128,512,8)
        out = self.dec5(out)#(-1,128,512,8)
        out = self.dec6(torch.cat([out, x], dim=1))
        return out
        #return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=4, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True))
        self.disc2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=4, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.disc3 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=4, padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True))


    def forward(self, x):
        out_d = self.disc1(x)
        out_d = self.disc2(out_d)
        out_d = self.disc3(out_d)
        #out_d = self.disc4(out_d)
        return out_d

def normalize(values, lower, upper):
    return (values - lower) / (upper - lower)

def denormalize(values, lower=0, upper=1):
    return values * (upper - lower) + lower
def to_grayscale(values, lower, upper):
    #normalized = normalize(values, lower, upper)
    #denormalized = np.clip(denormalize(normalized, 0, 255), 0, 255)
    #return denormalized.astype(np.float32)

    return values

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
    batch_size = 8
    ones = torch.ones(batch_size, 1, 8, 8).to(device)
    zeros = torch.zeros(batch_size, 1, 8, 8).to(device)
    # 損失関数
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()
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
            real_ips = real_ips[:,:,:512,:256].to(device)
            real_ips = torch.cat([real_ips[:,0,:,:],real_ips[:,1,:,:]],dim=2)
            real_ips = torch.reshape(real_ips, (-1, 1,512,512))
            #real_fdica = ips2fdica(real_ips)#シャッフル
            real_fdica = real_ips.clone()

            #確認用
            #spectrogram(real_ips[0,0,].cpu(),f"./Output/train_result/models/ips_{i:03}")
            #spectrogram(real_fdica[0,0,].cpu(),f"./Output/train_result/models/pp_{i:03}")

            # Gの訓練
            # 偽のカラー画像を作成
            fake_ips = model_G(real_fdica[:,:,:,:])#batch,ch,fight=4097,wideth=121?
            #fake_ips_fdica = torch.cat([real_fdica[:,:1,:,:], fake_ips], dim=1)
            #fake_ips_fdica = fake_ips#無変換

            # 偽画像を一時保存
            fake_ips_fdica_tensor = fake_ips.detach()

            # 偽画像を本物と騙せるようにロスを計算
            out = model_D(fake_ips)#batchsizwe,2,fight=4097,wideth=128?
            loss_G_bce = bce_loss(out, ones[:batch_len])#batch,1,128,2
            #print(loss_G_bce)

            ips_image1 = torch.cat([real_ips[:,:,:,:256],real_ips[:,:,:,256:]],dim=3)
            ips_image2 = torch.cat([real_ips[:,:,:,256:],real_ips[:,:,:,:256]],dim=3)

            e1 = mae_loss(fake_ips,ips_image1)#誤差１
            e2 = mae_loss(fake_ips,ips_image2)#誤差１
            e3 = mae_loss(ips_image1,ips_image2)#誤差１

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
            real_out = model_D(real_ips[:,:,:,:])
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
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

            spectrogram(real_ips[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real1_{i:03}")
            #spectrogram(real_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real2_{i:03}")
            spectrogram(fake_ips[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake1_{i:03}")
            #spectrogram(fake_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake2_{i:03}")
            spectrogram(real_fdica[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fdica1_{i:03}")
            #spectrogram(real_fdica[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fdica2_{i:03}")

    # ログの保存
    with open(f"./Output/train_result/models/epoch_{i:03}/logs.pickle", "wb") as fp:
        pickle.dump(result, fp)


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