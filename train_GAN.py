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
        file_num = glob.glob('./Output/fdiced_Z_spec/*')

        for i in tqdm(range(2**9)):
        #while len(self.data)<=2**10-1:#2**n個のデータセット
            #print(len(self.data))
            #load specs
            with open(random.choice(file_num), 'rb') as f:
                specs = pickle.load(f)#4097,121,3
            #mat化
            img_matrix = np.zeros([2,specs.shape[0],specs.shape[1]],dtype='float32')#[ch,heigh weight]
            img_matrix[0,:,:] = np.power(specs[:,:,0],2)
            img_matrix[1,:,:] = np.power(specs[:,:,1],2)
            #正規化?
            log_img_matrix = img_matrix
            gray_ips_matrix = to_grayscale(log_img_matrix, np.amin(log_img_matrix), np.amax(log_img_matrix))
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
        self.enc1 = self.conv_bn_relu(2, 32, kernel_size=5) # 32x96x96
        self.enc2 = self.conv_bn_relu(32, 64, kernel_size=3, pool_kernel=4)  # 64x24x24
        self.enc3 = self.conv_bn_relu(64, 128, kernel_size=3, pool_kernel=2)  # 128x12x12
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)  # 256x6x6

        self.dec1 = self.conv_bn_relu(256, 128, kernel_size=3, pool_kernel=-2)  # 128x12x12
        self.dec2 = self.conv_bn_relu(128 + 128, 64, kernel_size=3, pool_kernel=-2)  # 64x24x24
        self.dec3 = self.conv_bn_relu(64 + 64, 32, kernel_size=3, pool_kernel=-4)  # 32x96x96
        self.dec4 = nn.Sequential(
            nn.Conv2d(32 + 32, 2, kernel_size=5, padding=2),
            nn.Tanh()
        )

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        out = self.dec1(x4)
        out = self.dec2(torch.cat([out, x3], dim=1))
        out = self.dec3(torch.cat([out, x2], dim=1))
        out = self.dec4(torch.cat([out, x1], dim=1))
        return out

class Discriminator(nn.Module):
    # Inputの色空間はYCrCb→RGBにする
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_relu(2, 16, kernel_size=5, reps=1) # RGB
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=4)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=2)
        self.conv4 = self.conv_bn_relu(64, 128, pool_kernel=2)
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=2)
        self.out_patch = nn.Conv2d(256, 1, kernel_size=1) #1x3x3

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch,
                                    out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        y1 =self.conv1(x)
        y2 =self.conv2(y1)
        y3 =self.conv3(y2)
        y4 =self.conv4(y3)
        #y4 =self.conv4(x)
        out = self.conv5(y4)
        #out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return self.out_patch(out)

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
            flag =random.randint(0, 1)
            if flag ==1:
                real_fdica[batch_n,0,freq,:] = real_ips[batch_n,0,freq,:].copy()
                real_fdica[batch_n,1,freq,:] = real_ips[batch_n,1,freq,:].copy()
            elif flag==0:
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
    batch_size = 4
    ones = torch.ones(batch_size, 1, 128, 2).to(device)
    zeros = torch.zeros(batch_size, 1, 128, 2).to(device)
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
            real_ips = real_ips.to(device)
            real_fdica = ips2fdica(real_ips)#シャッフル
            #確認用
            #spectrogram(real_ips[0,0,].cpu(),f"./Output/train_result/models/ips_{i:03}")
            #spectrogram(real_fdica[0,0,].cpu(),f"./Output/train_result/models/pp_{i:03}")

            # Gの訓練
            # 偽のカラー画像を作成
            fake_ips = model_G(real_fdica[:,:,:4096,:64])#batch,ch,fight=4097,wideth=121?
            #fake_ips_fdica = torch.cat([real_fdica[:,:1,:,:], fake_ips], dim=1)
            #fake_ips_fdica = fake_ips#無変換

            # 偽画像を一時保存
            fake_ips_fdica_tensor = fake_ips.detach()

            # 偽画像を本物と騙せるようにロスを計算
            out = model_D(fake_ips)#batchsizwe,2,fight=4097,wideth=128?
            loss_G_bce = bce_loss(out, ones[:batch_len])#batch,1,128,2
            #print(loss_G_bce)

            e1 = mae_loss(fake_ips, real_ips[:,:,:4096,:64])#誤差１

            rev_real_ips = torch.zeros_like(fake_ips).to(device)
            tmp_mat = rev_real_ips.clone()
            rev_real_ips[:,0,:,:] = tmp_mat[:,1,:,:]
            rev_real_ips[:,1,:,:] = tmp_mat[:,0,:,:]
            rev_real_ips = rev_real_ips.to(device)
            e2 = mae_loss(fake_ips, rev_real_ips[:,:,:4096,:64])#誤差2
            #print('e1={},e2={}'.format(e1,e2))

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
            real_out = model_D(real_ips[:,:,:4096,:64])
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

        # 画像を保存
        if not os.path.exists("./Output/train_result"):
            os.mkdir("./Output/train_result")
        # 生成画像を保存
        #a = fake_ips_fdica_tensor[:1]
        #b = torch.cat([real_ips[:1],fake_ips_fdica_tensor[:1]],dim=3)
        #torchvision.utils.save_image(fake_ips_fdica_tensor[:1],
        #                        f"stl_color/fake_epoch_{i:03}.png")

        # モデルの保存
        if not os.path.exists(f"./Output/train_result/models/epoch_{i:03}/"):
            os.mkdir(f"./Output/train_result/models/epoch_{i:03}/")
        if i % 10 == 0 or i == 199:
            torch.save(model_G.state_dict(), f"./Output/train_result/models/epoch_{i:03}/gen_{i:03}.pytorch")
            torch.save(model_D.state_dict(), f"./Output/train_result/models/epoch_{i:03}/dis_{i:03}.pytorch")
            #torchvision.utils.save_image(real_ips[0,0,],f"./Output/train_result/models/real_epoch_{i:03}.png")


            #save_grayscale(real_ips[0,0,:,:],i)

            spectrogram(real_ips[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real1_{i:03}")
            spectrogram(real_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_real2_{i:03}")
            spectrogram(fake_ips[0,0,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake1_{i:03}")
            spectrogram(fake_ips[0,1,:,:].detach().cpu(),f"./Output/train_result/models/epoch_{i:03}/spec_fake2_{i:03}")

    # ログの保存
    with open("stl_color/logs.pkl", "wb") as fp:
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