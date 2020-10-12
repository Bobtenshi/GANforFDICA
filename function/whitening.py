import numpy as np
import wave
import math
from scipy.fftpack import fft, ifft
from scipy import hamming
from scipy.io import wavfile
from scipy.signal import resample
import scipy.linalg as LA
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






def newwhite(X,epsilon=1e-5):
    n,p,m=X.shape
    Z = np.zeros(X.shape, dtype=np.complex128)
    X1 = X[:,:,0]
    X2 = X[:,:,1]


    u,v = np.linalg.eig(np.dot(X1.T,X1)/n)  #u固有値 vベクトル
    Z1=np.dot(X1,np.dot(v,np.diag(1/(np.sqrt(u)+epsilon))))

    u,v = np.linalg.eig(np.dot(X2.T,X2)/n)  #u固有値 vベクトル
    Z2=np.dot(X2,np.dot(v,np.diag(1/(np.sqrt(u)+epsilon))))

    Z[:,:,0] = Z1
    Z[:,:,1] = Z2

    return (Z)