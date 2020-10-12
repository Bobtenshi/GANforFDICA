import numpy as np
import numpy.linalg as LA
from numpy import mat
from numpy.random import rand
from stft_or import stft, istft, whitening
from function.projection_back import projection_back
import numpy.matlib
from function import pps


def cost_function(P, R, W, I, J):
    A = np.zeros([I, 1])
    for i in range(I):
        x = np.abs(LA.det(W[:, :, i]))
        x = max(x, 1e-10)
        A[i] = 2 * np.log(x)
    return -J * A.sum() + (P / R + np.log(R)).sum()



# coding: utf-8


def FDICA(X , itr):
    """% [inputs]
%         X: observed multichannel spectrogram (freq. x time frames x channels)
%       itr: number of iterations (scalar)
%  drawCost: draw convergence behavior or not (true/false)
%
% [outputs]
%         Y: estimated signals (freq. x time frames x channels)
%         W: demixing matrix (source x channel x freq.) 
%
"""
    import librosa.display
    import pylab as plt
    import numpy as np
    from scipy import signal
    from sklearn.decomposition import PCA


    I,J,M = X.shape
    E = np.eye(M)
    W = np.zeros([M,M,I], dtype='complex')
    Y = np.zeros([I,J,M], dtype='complex')

    for i in range(I):
        W[:,:,i] = np.eye(M)
        Y[i,:,:] = X[i,:,:]
    Xp= np.transpose(X, (2, 1, 0) ) #M*J*I
    Yp= np.transpose(Y, (2, 1, 0) ) #M*J*I


    cost = 0


    print('Iteration:    ')
    for it in range(itr):

        for i in range(I):
            for m in range(M):
                rm = np.zeros([1,J], dtype='complex')
                for j in range(J):
                    rm[0,j]= max(  (abs(Yp[m,j,i])),100 * 2.2204e-16   ); # 1 x J
                #print(" rm =  "+str(rm.shape))

                dg = np.ones((M,1),dtype="complex") *  (1/rm)# M x J
                #print(" dg =  "+str(dg.shape))

                Vk = np.dot((dg*Xp[:,:,i]) , np.mat(Xp[:,:,i]).H  )/ J     #  M x M
                #print(" vk =  "+str(Vk.shape))

                A = np.dot(W[:,:,i],Vk)     #   M x M

                "疑似逆行列ver"
                A = A.I

                b  = E[:,m]         #  2 x 1
                b = np.reshape(b, (2, 1))

                wm = A * b
                wm =  wm  / np.sqrt((wm.H) * Vk * wm)

                Yp[m,:,i] = (wm.H)*Xp[:,:,i]
                W[m,:,i] = wm.H


    Y = np.transpose(Yp, (2, 1, 0) ) #M*J*I
    print(' FDICA done.\n')

    return Y,W


def bss_fdica(mix,origin,ns, nb, fftSize, shiftSize, it, type=1, draw=False, **params):
    """
    % [inputs]
    %        mix: observed mixture (len x mic)
    %         ns: number of sources (scalar)
    %         nb: number of bases (scalar)
    %    fftSize: window length in STFT (scalar)
    %  shiftSize: shift length in STFT (scalar)
    %         it: number of iterations (scalar)
    %       type: 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
    %       draw: plot cost function values or not (logic, true or false)
    """

    # Short-time Fourier transform
    X, window = stft(mix,fftSize,shiftSize)
    S, window = stft(origin,fftSize,shiftSize)
    I,J,M = S.shape
    # Whiteing (applying PCA)
    #Xwhite = whitening(X, ns) # decorrelate input multichannel signal

    # ILRMA
    #Y, e = ilrma(X, Xwhite, type, it, nb, draw, **params)
    itr = 10
    Y,estW= FDICA(X,itr)
    #print(type(Y))
    #print(type(Y_r))
    #X = np.random.rand(513, 11)  # 4 x 4の配列の乱数
    estY,D= projection_back(Y, X[:,:,0])
    #Z=np.array(Z)

    pW =pps.perfectPermuSolver(estW,estY,S)
    #pW = pW.detach().numpy()

    Yp = np.zeros([2,J,I], dtype='complex')
    Xwp = np.transpose(X, (2, 1, 0) )

    for i in range(I): # separation
        #Yp[:,:,i]= pW[:,:,i] * Xwp[:,:,i]
        Yp[:,:,i]= np.dot(pW[:,:,i],Xwp[:,:,i])

    Y = np.transpose(Yp, (2, 1, 0) )
    Z,D= projection_back(Y, X[:,:,0])

    est_z = istft(Z, shiftSize, window, mix.shape[0])

    #return sep, e



    return est_z
