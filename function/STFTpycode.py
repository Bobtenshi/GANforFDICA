#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import hamming
import soundfile as sf
from scipy.signal import resample,resample_poly
import scipy.linalg as LA
import pylab as pl
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import os
import librosa.display

"""
%% 関数一覧 %%
% def stft(signal, fftSize=2048, shiftSize=1024, window='hamming'):
% def optSynWnd(analysisWindow, shiftSize):#逆窓関数を導出
% def istft(S, shiftSize,sumple_len,window=None):#istft code

% def spectrogram(S, output_path=None, sampling_rate=8000, fftSize=None, shiftSize=None, **kwargs):
% def showspect(S):
"""

### for STFT ###

def stft(signal, fftSize=2048, shiftSize=1024, window='hamming'):
    """
    Parameters
    ----------
    signal: input signal
    fftSize: frame length
    shiftSize: frame shift
    window: window function(hamming,hanning,bartlett,blackman)

    Returns
    -------
    S: spectrogram of input signal (fftSize/2+1 x frame x ch)
    window: used window function (fftSize x 1)
    """
    signal = np.array(signal)#singnal 2 numpy array
    if window == 'hamming':#ハミング窓
        window = hamming(fftSize+1)[:fftSize]
    elif window=='hanning':#ハニング窓
        window =hanning(fftSize+1)[:fftSize]
    elif window=='bartlett':#バートレット窓
        window =bartlett(fftSize+1)[:fftSize]
    elif window=='blackman':#ブラックマン窓
        window =blackman(fftSize+1)[:fftSize]

    zeroPadSize = fftSize - shiftSize#0詰め　信号序盤
    length = signal.shape[0]#signal サンプリング点数
    frames = int(np.floor((length + fftSize - 1) / shiftSize))#取得フレーム数
    I = int(fftSize/2 + 1)#ナイキスト数

    if len(signal.shape) == 1:#ch=1 monoral
        signal = np.concatenate([np.zeros(zeroPadSize), signal, np.zeros(fftSize)])#0詰め処理
        S = np.zeros([I, frames], dtype=np.complex128)#スペクトル格納メモリ　確保

        for j in range(frames):#
            start_posi = j * shiftSize#分割位置
            spectrum = fft(signal[ start_posi:  start_posi+fftSize] * window)#fft(signal*窓関数)
            S[:, j] = spectrum[:I]#スペクトル格納

        return S, window,length#スペクトル,使用窓関数（istftの為）

    elif len(signal.shape) >= 2:##ch=2 stereo
        nch = signal.shape[1]#
        signal = np.concatenate([np.zeros([zeroPadSize, nch]), signal, np.zeros([fftSize,nch])])#
        S = np.zeros([I, frames, nch], dtype=np.complex128)#

        for ch in range(nch):#
            for j in range(frames):#
                sp = j * shiftSize#
                spectrum = fft(signal[sp: sp+fftSize, ch] * window)#
                S[:, j, ch] = spectrum[:I]#

        return S, window,length#スペクトル,使用窓関数（istftの為）

    else:#ch=2まで対応　３ch~ 対応必要
        raise ValueError('illegal signal dimension')#error

def spectrogram(S, output_path=None, sampling_rate=8000, fftSize=None, shiftSize=None, **kwargs):
    """
    スペクトログラム表示関数
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable#
    I, J = S.shape#

    if fftSize is None:#
        fftSize = (I - 1) * 2#

    if shiftSize is None:#
        shiftSize = fftSize // 2#

    X, Y = pl.meshgrid(#
        (pl.arange(J + 1) - 0.5) * shiftSize / sampling_rate,  # sec
        (pl.arange(I + 1) - 0.5) / fftSize * sampling_rate  # Hz
    )

    fig, ax = plt.subplots()#

    image = ax.pcolormesh(X, Y, np.log(S + 1e-5), **kwargs)#
    divider = make_axes_locatable(ax)#
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)#
    fig.add_axes(ax_cb)#
    plt.xlabel('time [s]', fontsize='x-large')#
    plt.ylabel('frequency [Hz]', fontsize='x-large')#
    plt.xticks(fontsize='large')#
    plt.yticks(fontsize='large')#
    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)#
    plt.colorbar(image, cax=ax_cb)#


def showspect(S):
    import librosa.display
    """
    スペクトログラム表示関数
    """
    S = np.log(np.abs(S) ** 2)   #対数取得
    # 対数パワーを、横軸時間・縦軸周波数で表示
    librosa.display.specshow(S,
                             x_axis='time',
                             y_axis='hz',
                             cmap='magma')
    
    
    
    plt.title('stft Power spectrogram')    # 図のタイトル設定
    plt.colorbar(format='%+2.0f dB')  # 色とデシベル値の対応図を表示
    plt.colorbar()
    plt.tight_layout()                # 余白を小さく
    plt.show()                        # 図を表示する

def optSynWnd(analysisWindow, shiftSize):#逆窓関数を導出
    fftSize = analysisWindow.shape[0]#
    synthesizedWindow = np.zeros(fftSize)#
    for i in range(shiftSize):#
        amp = 0#
        for j in range(1, int(fftSize / shiftSize) + 1):#
            amp += analysisWindow[i + (j-1) * shiftSize] ** 2#
        for j in range(1, int(fftSize / shiftSize) + 1):#
            synthesizedWindow[i + (j-1) * shiftSize] = analysisWindow[i + (j-1) * shiftSize] / amp#

    return synthesizedWindow#

def istft(S, shiftSize,sumple_len,window=None):#istft code
    """
    % [inputs]
    %           S: STFT of input signal (fftSize/2+1 x frames x nch)
    %   shiftSize: frame shift (default: fftSize/2)
    %      window: window function used in STFT (fftSize x 1) or choose used
    %              function from below.
    %              'hamming'    : Hamming window (default)
    %              'hann'       : von Hann window
    %              'rectangular': rectangular window
    %              'blackman'   : Blackman window
    %              'sine'       : sine window
    %      length: length of original signal (before STFT)
    %
    % [outputs]
    %   waveform: time-domain waveform of the input spectrogram (length x nch)
    %
    """
    if window is None:#窓関数指定なし＝ハミング窓使用
        fftSize = shiftSize * 2#
        window = hamming(fftSize+1)[:fftSize]#

    if S.ndim == 2:#monoral
        freq, frames = S.shape# 縦＝周波数軸数 横＝フレーム数
        fftSize = (freq-1) * 2#本来の周波数軸分
        invWindow = optSynWnd(window, shiftSize)#逆窓関数の導出
        spectrum = np.zeros(fftSize, dtype=np.complex128)#メモリ確保　1＊fftサイズ分

        tmpSignal = np.zeros([(frames - 1) * shiftSize + fftSize])#ifft　deta　格納
        for j in range(frames):#全スペクトグラム　run
            spectrum[:int(fftSize / 2) + 1] = S[:, j]#すべての行のj番目＝1つのfftベクトル
            spectrum[0] /= 2#直流成分　1/2
            spectrum[int(fftSize / 2)] /= 2#？？
            sp = j * shiftSize#
            tmpSignal[sp: sp + fftSize] += (np.real(ifft(spectrum, fftSize) * invWindow) * 2)#ifftして重ね足し
        #signal = np.concatenate([np.zeros(zeroPadSize), signal, np.zeros(fftSize)])#0詰め処理
        #waveform = tmpSignal[fftSize - shiftSize: (frames - 1) * shiftSize + fftSize]#0詰め考慮
        waveform = tmpSignal[fftSize - shiftSize:fftSize - shiftSize + sumple_len]#0詰め考慮

    elif S.ndim >= 3:#
        freq, frames, nch = S.shape#
        fftSize = (freq-1) * 2#
        invWindow = optSynWnd(window, shiftSize)#
        spectrum = np.zeros(fftSize, dtype=np.complex128)#

        tmpSignal = np.zeros([(frames - 1) * shiftSize + fftSize, nch])#
        for ch in range(nch):#
            for j in range(frames):#
                spectrum[:int(fftSize / 2) + 1] = S[:, j, ch]#
                spectrum[0] /= 2#
                spectrum[int(fftSize / 2)] /= 2#
                sp = j * shiftSize#
                tmpSignal[sp: sp + fftSize, ch] += (np.real(ifft(spectrum, fftSize) * invWindow) * 2)#

        waveform = tmpSignal[fftSize - shiftSize: fftSize - shiftSize + sumple_len]#

    #if length:#
        #waveform = waveform[:length]#

    return waveform #
