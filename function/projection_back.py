import numpy as np


def projection_back(Y, X):
    """
    % This function restores the scale of the signals by estimated by ICA-based
    % blind source separation techniques.
    %
    % see also
    % http://d-kitamura.net
    %
    % [inputs]
    %   Y: estimated (separated) signals (freq. x frames x sources)
    %   X: observed (mixture) signal with desired channel (freq. x frames x 1)
    %      or observed multichannel signals (freq. x frames x channels)
    %
    """
    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]
    assert Y.ndim == 3

    I, J, M = Y.shape

    if X.ndim == 2:
        A = np.zeros([M, I], dtype=np.complex128)
        D = np.zeros([M,M,I], dtype=np.complex128)
        Z = np.zeros(Y.shape, dtype=np.complex128)
        for i in range(I):
            Yi = np.mat(Y[i, :, :]).T  # (M, J)
            A[:, i] = np.mat(X[i, :]) * Yi.H * (Yi * Yi.H).I  # (J) * (J, M) * (M, M)

        for m in range(M): # todo m loopはなくせるはず
            for i in range(I):
                Z[i, :, m] = A[m, i] * Y[i, :, m]
                D[m, m, i] = A[m, i]

    elif X.ndim == 3:
        A = np.zeros([M, M, I], dtype=np.complex128)
        Z = np.zeros([I, J, M, M], dtype=np.complex128)  # (I, J, N, M)
        for i in range(I):
            for m in range(M):
                Yi = np.mat(Y[i, :, :]).T  # (M, J)
                A[m, :, i] = np.mat(X[i, :, m]) * Yi.H * (Yi * Yi.H).I  # (1, J) * (J, M) * (M, M)

        for n in range(M):
            for m in range(M):
                for i in range(I):
                    Z[i, :, n, m] = A[m, n, i] * Y[i, :, n]  # scalar * (J)

        D = A

    else:
        raise ValueError('X.ndim must be 2 or 3')

    return Z, D
