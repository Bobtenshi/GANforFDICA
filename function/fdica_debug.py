
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
    W = np.zeros([M,M,I], dtype='float')
    Y = np.zeros([I,J,M], dtype='float')
    
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
                rm = np.zeros([1,J], dtype='float')
                for j in range(J):
                    rm[0,j]= max(  (abs(Yp[m,j,i])),100 * 2.2204e-16   ); # 1 x J
                #print(" rm =  "+str(rm.shape))
                
                dg = np.ones((M,1),dtype="float") *  (1/rm)# M x J
                #print(" dg =  "+str(dg.shape))
                
                Vk = np.dot((dg*Xp[:,:,i]) , np.mat(Xp[:,:,i]).H  )/ J     #  M x M
                #print(" vk =  "+str(Vk.shape))

                A = np.dot(W[:,:,i],Vk)     #   M x M

                "疑似逆行列ver"
                A = A.I
                #A = np.linalg.pinv(A)

                #print(" A =  "+str(A.shape))
                b  = E[:,m]         #  2 x 1
                b = np.reshape(b, (2, 1))
                #print(" b =  "+str(b.shape))
                #print(b)
                #Q, R = np.linalg.qr()
                #t = np.dot(Q.T, b)
                #wm = np.linalg.solve(R, t)
                wm = A * b
                wm =  wm  / np.sqrt((wm.H) * Vk * wm)
                
                Yp[m,:,i] = (wm.H)*Xp[:,:,i]
                W[m,:,i] = wm.H
            

    Y = np.transpose(Yp, (2, 1, 0) ) #M*J*I
    print(' FDICA done.\n')
    
    return Y,W


if __name__ == "__main__":
    itr = 10

    X = 
    Y,W = FDICA(X , itr)
