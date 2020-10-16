#-----------------------------------------
# import
#-----------------------------------------
import numpy as np
import wave
import math
from scipy.fftpack import fft, ifft
from scipy import hamming
from scipy.io import wavfile
from scipy.signal import resample
import scipy.linalg as LAif
from Toolbox import stft_or as stft
import pickle
#-----------------------------------------
#functions
#-----------------------------------------
def fake_fdica(nch,data_n):
  #引数 ch数
  #return S,Z
  #S　源信号　(f,j,ch)
  #Z　疑似FDICA＆IPS (f,j,ch)

  for n in range(data_n):
    #wav参照
    #nch = 2
    dry_signal_mat = np.zeros([240000, nch])
    for m in range(nch):
      target_wav_n = np.random.randint(m*(100//nch), (m+1)*(100//nch))
      fs, signal= wavfile.read('Input/input_source_10s/'+str(target_wav_n)+'.wav')
      dry_signal_mat[:,m] = signal / 32768.0

    #write wav file
    #wavfile.write('ica_signal1.wav', 16000, mix_siganal_mat[:,0])
    #wavfile.write('ica_signal2.wav', 16000, mix_siganal_mat[:,1])
    #wavfile.write('ica_signal3.wav', 16000, mix_siganal_mat[:,2])

    #STFT & fake fdica prosses
    fftSize=2*1024
    shiftSize=512
    S, window = stft.stft(dry_signal_mat, fftSize, shiftSize)
    Z = np.zeros_like(S)

    for freq in range(S.shape[0]):#S[4096,89,3]
      for aim_ch in range(nch):#対象ch数
        temp_est_activation = 0
        main_rate = np.random.normal(0.98,0.02,1)
        sub_rate = (1-main_rate)/(nch-1)

        for m in range(nch):#前ch数走査
          if m == aim_ch:
            temp_est_activation += S[freq,:,m]*main_rate
          else:
            temp_est_activation += S[freq,:,m]*sub_rate

        Z[freq,:,aim_ch] = temp_est_activation
      #print("fake FDICA at freq {} is done ".format(freq))

    #print("{}ch fake-FDICA spectrogram is cooked".format(nch))

    with open(f'./Output/fdiced_Z_spec/{nch:01}_ch/Z_{n:01}.pickle', 'wb') as f:
      pickle.dump(Z,f)
    f.close
    if n%10==0:
      print("{}files {}ch fake-FDICAed spectrogram is cooked.".format(n,nch))
    #return S,Z

def sum(a,b):
  c = a + b
  return c
#-----------------------------------------
# main
#-----------------------------------------
if __name__ == '__main__':

  fake_fdica(nch=2,data_n=100)
  #print(S)
  #print(Z)

