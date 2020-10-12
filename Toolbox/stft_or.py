#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import wave
import math
from scipy.fftpack import fft, ifft
from scipy import hamming
from scipy.io import wavfile
from scipy.signal import resample
import scipy.linalg as LA

### for STFT ###

def stft(signal, fftSize, shiftSize, window='hamming'):
    """
    Parameters
    ----------
    signal: input signal
    fftSize: frame length
    shiftSize: frame shift
    window: window function

    Returns
    -------
    S: spectrogram of input signal (fftSize/2+1 x frame x ch)
    window: used window function (fftSize x 1)
    """
    signal = np.array(signal)

    if window == 'hamming':
        # todo 色々対応
        window = hamming(fftSize+1)[:fftSize]

    nch = signal.shape[1]

    zeroPadSize = fftSize - shiftSize
    signal = np.concatenate([np.zeros([zeroPadSize, nch]), signal, np.zeros([fftSize,nch])])
    length = signal.shape[0]

    frames = int(np.floor((length - fftSize + shiftSize) / shiftSize))
    I = int(fftSize/2 + 1)
    S = np.zeros([I, frames, nch], dtype=np.complex128)

    for ch in range(nch):
        for j in range(frames):
            sp = j * shiftSize
            spectrum = fft(signal[sp: sp+fftSize, ch] * window)
            S[:, j, ch] = spectrum[:I]

    return S, window


def whitening(X, dnum=2):
    # M == dnumで固定 (todo)
    I, J, M = X.shape
    Y = np.zeros(X.shape, dtype=np.complex128)

    def _whitening(Xi):
        V = Xi @ Xi.T.conjugate() / J # covariance matrix (M, M)
        eig_val, P = LA.eig(V)
        D = np.diag(eig_val)

        idx = np.argsort(eig_val)
        D = D[idx, idx]
        P = P[:, idx]
        return (np.diag(D ** (-0.5)) @ P.T.conjugate() @ Xi).T # (M, M) * (M, M) * (M, J)

    for i in range(I):
        Y[i] = _whitening(X[i].T)

    return Y


def power(S):
    return np.real(S) ** 2 + np.imag(S) ** 2


def spectrogram(S, output_path=None):
    import pylab as pl
    I, J = S.shape
    X, Y = pl.meshgrid(pl.arange(J+1), pl.arange(I+1))
    pl.pcolor(X, Y, np.log(S))
    if output_path is None:
        pl.show()
    else:
        pl.savefig(output_path)

### for ISTFT ###

def optSynWnd(analysisWindow, shiftSize):
    fftSize = analysisWindow.shape[0]
    synthesizedWindow = np.zeros(fftSize)
    for i in range(shiftSize):
        amp = 0
        for j in range(1, int(fftSize / shiftSize) + 1):
            amp += analysisWindow[i + (j-1) * shiftSize] ** 2
        for j in range(1, int(fftSize / shiftSize) + 1):
            synthesizedWindow[i + (j-1) * shiftSize] = analysisWindow[i + (j-1) * shiftSize] / amp

    return synthesizedWindow


def istft(S, shiftSize, window, length):
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
    freq, frames, nch = S.shape
    fftSize = (freq-1) * 2
    invWindow = optSynWnd(window, shiftSize)

    spectrum = np.zeros(fftSize, dtype=np.complex128)
    tmpSignal = np.zeros([(frames - 1) * shiftSize + fftSize, nch])
    for ch in range(nch):
        for j in range(frames):
            spectrum[:int(fftSize/2)+1] = S[:, j, ch]
            spectrum[0] /= 2
            spectrum[int(fftSize/2)] /= 2
            sp = j * shiftSize
            tmpSignal[sp: sp+fftSize, ch] += (np.real(ifft(spectrum, fftSize) * invWindow) * 2)

    waveform = tmpSignal[fftSize-shiftSize: (frames-1)*shiftSize+fftSize]

    waveform = waveform[:length]
    return waveform


def main(input_path, output_path):
    """
    input_path : wav
    output_path : png
    """
    fs, signal = wavfile.read(input_path)
    S, window = stft(signal, fftSize=4096*2, shiftSize=2048)
    S = power(S)
    spectrogram(S[:, :, 0], output_path)



if __name__ == '__main__':
    # exit(main('input/synth_res.wav', 'test.png'))

    fs, signal1 = wavfile.read('input/guitar_res.wav')
    signal1 = signal1 / 32768.0
    assert fs == 16000
    fs, signal2 = wavfile.read('input/synth_res.wav')
    signal2 = signal2 / 32768.0
    assert fs == 16000
    nch = 2

    mix = np.zeros([len(signal1), nch])

    for ch in range(nch):
        mix[:, ch] = signal1[:, ch] + signal2[:, ch]

    S, window = stft(mix, fftSize=4096, shiftSize=2048)
    mix2 = istft(S, 2048, window, mix.shape[0])
    wavfile.write('est.wav', 16000, mix2)

