import numpy as np
import numpy.linalg as LA
from numpy import mat
from numpy.random import rand

from mir_eval.separation import bss_eval_sources
import sys, os
from scipy.io import wavfile
from stft import stft, istft, whitening
from separation.projection_back import projection_back

import chainer.functions as F
from dnn.make_datasets import format_vector, c

from dnn.model import predict as mpredict
from dnn.evaluate import format_data
from dnn.divergence import itakura_saito_divergence
from stft import spectrogram
import notify
from separation.evaluater import Evaluate

na = np.array


def cost_function(P, R, W, I, J):
    A = np.zeros([I, 1])
    for i in range(I):
        x = np.abs(LA.det(W[:, :, i]))
        x = max(x, 1e-10)
        A[i] = 2 * np.log(x)
    return -J * A.sum() + (P / R + np.log(R)).sum()


def modify_permutation(U, w, i):
    """

    Args:
        U: U_{i} N * M * M
        w: w_{i} N * M

    Returns:
        w: w_{i} N * M
    """
    N, M = w.shape
    if N == 2:
        normal = np.real(w[0].T.conjugate() @ U[0] @ w[0] + w[1].T.conjugate() @ U[1] @ w[1])
        inv = np.real(w[0].T.conjugate() @ U[1] @ w[0] + w[1].T.conjugate() @ U[0] @ w[1])
        if normal > inv:
            # print(i, end=',')
            w = w[[1, 0]]
    else:
        raise NotImplementedError

    return w


def idlma(Xinput, models, It, draw, randomly_initW, refMic, dump_step, n_update_w, e, dump_spectrogram, delta, wf_type, delta_type, **params):
    """
     [inputs]
          X: I * J * M
          models: type(list) DNN model each instruments
          It: number of iterations
          randomly_initW : bool
          delta: 大きくすればするほどDNNの出力を曖昧にできる(?)
          dump_spectrogram: bool
          e: 評価インスタンス
          wf_type: 0 無し, 1 それぞれ, 2 もう一方の残差を足す, 3 inputにWF
          delta_type: "add" if r = r + δ,  "max" if r = max(r, δ)

     [outputs]
          Y: estimated signals in time-frequency domain (freq. x frames x sources)
       cost: values of cost function in each iteration (It+1 x 1)
          W: demixing matrices (channels x channels x freq.)

    """
    print('delta: {}, wf_type: {}, delta_type: {}'.format(delta, wf_type, delta_type))
    I, J, M = Xinput.shape
    N = len(models)

    W = np.zeros([N, M, I], dtype=np.complex128)
    if randomly_initW:
        # 複素正規分布
        for i in range(I): W[:, :, i] = np.random.randn(N, M) * np.exp(np.random.rand(N, M) * 2 * np.pi * 1j)
    else:
        for i in range(I): W[:, :, i] = np.eye(N)

    # Initialization
    R = rand(I, J, N)
    Y = np.zeros([I, J, N], dtype=np.complex128)
    U = np.zeros([I, N, M, M], dtype=np.complex128) # U_{i,n}が M * Mの正方行列

    # spectrogram(np.abs(Xinput[:, :, refMic]), output_path='spec_org.png')
    for n in range(N):
        Y[:, :, n] = Xinput[:, :, refMic]

    costs = np.zeros([It+1,1])

    # initial
    e.eval_loss_func(np.abs(Y))
    e.eval_mir(Y)
    if dump_spectrogram:
        e.dump_spectrogram()

    print('Iteration: ')
    for it in range(1, It+1):

        if not (it - 1) % n_update_w:
            """ Update R """
            if wf_type == 0:
                for n in range(N):
                    R[:, :, n] = mpredict(models[n], Y[:, :, n]).data.T ** 2

            elif wf_type == 1:
                for n in range(N):
                    # winner filter
                    pred0 = mpredict(models[0], Y[:, :, n]).data.T ** 2 + 1e-5
                    pred1 = mpredict(models[1], Y[:, :, n]).data.T ** 2 + 1e-5

                    if n == 0:
                        R[:, :, 0] = np.abs(Y[:, :, 0]) * (pred0 / (pred0 + pred1)) ** 2
                    elif n == 1:
                        R[:, :, 1] = np.abs(Y[:, :, 1]) * (pred1 / (pred0 + pred1)) ** 2

            elif wf_type == 2:
                # multi winner filter
                s00 = mpredict(models[0], Y[:, :, 0]).data.T ** 2 + 1e-5
                s01 = mpredict(models[0], Y[:, :, 1]).data.T ** 2 + 1e-5
                s10 = mpredict(models[1], Y[:, :, 0]).data.T ** 2 + 1e-5
                s11 = mpredict(models[1], Y[:, :, 1]).data.T ** 2 + 1e-5
                R[:, :, 0] = np.abs(Y[:, :, 0] * (s00 / (s00 + s10)) + Y[:, :, 1] * (s01 / (s01 + s11))) ** 2
                R[:, :, 1] = np.abs(Y[:, :, 0] * (s10 / (s00 + s10)) + Y[:, :, 1] * (s11 / (s01 + s11))) ** 2

            elif wf_type == 3:
                # multi winner filter
                s00 = mpredict(models[0], Y[:, :, 0]).data.T ** 2 + 1e-5
                s01 = mpredict(models[0], Y[:, :, 1]).data.T ** 2 + 1e-5
                s10 = mpredict(models[1], Y[:, :, 0]).data.T ** 2 + 1e-5
                s11 = mpredict(models[1], Y[:, :, 1]).data.T ** 2 + 1e-5
                R[:, :, 0] = (np.abs(Y[:, :, 0] + Y[:, :, 1]) * ( (s00 + s01) / (s00 + s01 + s10 + s11) ) ) ** 2
                R[:, :, 1] = (np.abs(Y[:, :, 0] + Y[:, :, 1]) * ( (s10 + s11) / (s00 + s01 + s10 + s11) ) ) ** 2

            else:
                raise ValueError

            for n in range(N):
                _delta = R[:, :, n].mean() * delta
                if delta_type == 'add':
                    R[:, :, n] = R[:, :, n] + _delta
                elif delta_type == 'max':
                    R[:, :, n] = np.maximum(R[:, :, n], _delta)
                else:
                    raise ValueError

                if dump_spectrogram:
                    spectrogram(R[:, :, n], './output/new/_wf_{}_{}.png'.format(n, it), vmin=-5)

        for n in range(N):
            """ Update W """
            e_n = np.array([1 if _n == n else 0 for _n in range(N)])
            for i in range(I):
                U[i, n] = (np.matlib.repmat(1 / R[i, :, n], N, 1) * Xinput[i].T.conjugate()) @ Xinput[i] / J
                w = LA.solve(W[:, :, i] @ U[i, n], e_n)
                W[n, :, i] = (w / np.sqrt(w.T.conjugate() @ U[i, n] @ w)).T.conjugate()

        # if not (it - 10) % n_update_w:
        #     for i in range(I):
        #         W[:, :, i] = modify_permutation(U[i], W[:, :, i], i)

        for n in range(N):
            for i in range(I):
                Y[i, :, n] = W[n, :, i].T.conjugate() @ Xinput[i, :, :].T

        P = np.abs(Y) ** 2

        cost = cost_function(P, R, W, I, J)
        if draw:
            costs[it, 0] = cost

        verbose = False
        if verbose:
            print(it, cost)

        """ projection back """
        Z, D = projection_back(Y, Xinput[:, :, refMic])

        Y = Z
        for i in range(I):
            W[:, :, i] = D[:, :, i] @ W[:, :, i]  # update W

        if not it % dump_step:
            e.eval_loss_func(np.abs(Y), it)
            # e.dump_wav(Y, it)
            if dump_spectrogram:
                e.dump_spectrogram(Y, it)
            e.eval_mir(Y)

        assert Xinput.shape == (I, J, M)

    e.dump_sdrplot()
    print('Done')
    return Y, e


def idlma_semi_supervised(Xinput, models, It, draw, randomly_initW, refMic, dump_step, n_update_w, e, dump_spectrogram, delta, **params):
    """
     [inputs]
          X: I * J * M
          models: type(list) DNN model each instruments
          It: number of iterations
          randomly_initW : bool
          delta: 大きくすればするほどDNNの出力を曖昧にできる(?)
          dump_spectrogram: bool
          e: 評価インスタンス

     [outputs]
          Y: estimated signals in time-frequency domain (freq. x frames x sources)
       cost: values of cost function in each iteration (It+1 x 1)
          W: demixing matrices (channels x channels x freq.)

    """
    print('delta:', delta)
    I, J, M = Xinput.shape
    N = len(models)

    W = np.zeros([N, M, I], dtype=np.complex128)
    if randomly_initW:
        # 複素正規分布
        for i in range(I): W[:, :, i] = np.random.randn(N, M) * np.exp(np.random.rand(N, M) * 2 * np.pi * 1j)
    else:
        for i in range(I): W[:, :, i] = np.eye(N)

    # Initialization
    Y = np.zeros([I, J, N], dtype=np.complex128)
    L = 20  # number of bases
    T = rand(I, L, N)
    V = rand(L, J, N)
    R = np.zeros([I, J, N])
    for n in range(N):
        R[:, :, n] = T[:, :, n] @ V[:, :, n]  # R[:, :, 0] is unused

    U = np.zeros([I, N, M, M], dtype=np.complex128) # U_{i,n}が M * Mの正方行列

    # spectrogram(np.abs(Xinput[:, :, refMic]), output_path='spec_org.png')
    for n in range(N):
        Y[:, :, n] = Xinput[:, :, refMic]

    P = np.abs(Y) ** 2

    costs = np.zeros([It+1,1])

    # initial
    e.eval_loss_func(np.abs(Y))
    e.eval_mir(Y)
    if dump_spectrogram:
        e.dump_spectrogram()

    print('Iteration: ')
    for it in range(1, It+1):
        for n in range(N):
            """ Update R """
            if n == 1:  # vocals
                # estimate with DNN
                if not (it - 1) % n_update_w:
                    R[:, :, n] = mpredict(models[n], Y[:, :, n]).data.T
                    R[:, :, n] = R[:, :, n] ** 2 + delta

            else: # bass
                # estimate with NMF
                """ Update T """
                T[:,:,n] = T[:,:,n] * ( np.sqrt( (P[:,:,n] * (R[:,:,n] ** (-2))) @ V[:,:,n].T / ( (R[:,:,n] ** (-1)) @ V[:,:,n].T ) ) )
                T[:,:,n] = np.maximum(T[:,:,n], 1e-15)
                R[:,:,n] = T[:,:,n] @ V[:,:,n]

                """ Update V """
                V[:,:,n] = V[:,:,n] * ( np.sqrt( T[:,:,n].T @ (P[:,:,n] * (R[:,:,n] ** (-2))) / ( T[:,:,n].T @ (R[:,:,n] ** (-1)) ) ) )
                V[:,:,n] = np.maximum(V[:,:,n], 1e-15)
                R[:,:,n] = T[:,:,n] @ V[:,:,n]
                R[:, :, n] += delta

        for n in range(N):
            """ Update W """
            e_n = np.array([1 if _n == n else 0 for _n in range(N)])
            for i in range(I):
                U[i, n] = (np.matlib.repmat(1 / R[i, :, n], N, 1) * Xinput[i].T.conjugate()) @ Xinput[i] / J
                w = LA.solve(W[:, :, i] @ U[i, n], e_n)
                W[n, :, i] = (w / np.sqrt(w.T.conjugate() @ U[i, n] @ w + 1e-10)).T.conjugate()

        # if not (it - 10) % n_update_w:
        #     for i in range(I):
        #         W[:, :, i] = modify_permutation(U[i], W[:, :, i], i)

        for n in range(N):
            for i in range(I):
                Y[i, :, n] = W[n, :, i].T.conjugate() @ Xinput[i, :, :].T

        P = np.abs(Y) ** 2

        cost = cost_function(P, R, W, I, J)
        if draw:
            costs[it, 0] = cost

        print(it, cost)
        if cost > 1e+10:  # 発散
            print('=========STOP ITERATION=========')
            notify.main('[bss_idlma.py] {} is stopped at {}'.format(e.song_num, it))
            return Y, e

        """ projection back """
        Z, D = projection_back(Y, Xinput[:, :, refMic])

        Y = Z
        for i in range(I):
            W[:, :, i] = D[:, :, i] @ W[:, :, i]  # update W

        if not it % dump_step:
            e.eval_loss_func(np.abs(Y))
            # e.dump_wav(Y, it)
            if dump_spectrogram:
                e.dump_spectrogram(Y, it)
            e.eval_mir(Y)

        assert Xinput.shape == (I, J, M)

    e.dump_sdrplot()
    print('Done')
    return Y, e


# def _idlma_semi_supervised(X, models, It, draw, W, refMic, dump_step, n_update_w, e, **params):
#     """
#     ex)
#     vocal -> DNNで推定
#     bass -> NMFで推定
#
#      [inputs]
#           X: I * J * M
#           Xwhite: I * J * M
#           models: type(list) DNN model each instruments
#           It: number of iterations
#
#      [outputs]
#           Y: estimated signals in time-frequency domain (freq. x frames x sources)
#        cost: values of cost function in each iteration (It+1 x 1)
#           W: demixing matrices (channels x channels x freq.)
#
#     """
#     delta = 1e-3
#     Xinput = np.array(Xinput)  # これいる？
#
#     I, J, M = Xinput.shape
#     N = M
#     assert len(models) < N, 'モデルの数が足りています。半教師にする余地がありません。'
#
#     if W is None:
#         W = np.zeros([N, M, I], dtype=np.complex128)
#         for i in range(I): W[:, :, i] = np.eye(N)
#
#     # Initialization
#     Y = np.zeros([I, J, N], dtype=np.complex128)
#     L = 20  # number of bases
#     T = rand(I, L, N)
#     V = rand(L, J, N)
#     R = np.zeros([I, J, N])
#     for n in range(N):
#         R[:, :, n] = T[:, :, n] @ V[:, :, n]  # R[:, :, 0] is unused
#
#     for n in range(N):
#         Y[:, :, n] = Xinput[:, :, refMic]
#
#     costs = np.zeros([It+1,1])
#     P = np.abs(Y) ** 2
#
#     # Iterate
#     print('Iteration: ')
#     for it in range(1, It+1):
#         for n in range(N):
#             """ Update R """
#             if n == 0:
#                 # estimate with DNN
#                 if not (it - 1) % n_update_w:
#                     R[:, :, n] = mpredict(models[n], Y[:, :, n]).data.T
#                     R[:, :, n] = R[:, :, n] ** 2 + delta
#
#             else:
#                 # estimate with NMF
#                 """ Update T """
#                 T[:,:,n] = T[:,:,n] * ( np.sqrt( (P[:,:,n] * (R[:,:,n] ** (-2))) @ V[:,:,n].T / ( (R[:,:,n] ** (-1)) @ V[:,:,n].T ) ) )
#                 T[:,:,n] = np.maximum(T[:,:,n], 1e-15)
#                 R[:,:,n] = T[:,:,n] @ V[:,:,n]
#
#                 """ Update V """
#                 V[:,:,n] = V[:,:,n] * ( np.sqrt( T[:,:,n].T @ (P[:,:,n] * (R[:,:,n] ** (-2))) / ( T[:,:,n].T @ (R[:,:,n] ** (-1)) ) ) )
#                 V[:,:,n] = np.maximum(V[:,:,n], 1e-15)
#                 R[:,:,n] = T[:,:,n] @ V[:,:,n]
#
#             """ Update W """
#             e_n = np.array([1 if _n == n else 0 for _n in range(N)])
#             for i in range(I):
#                 D = (np.matlib.repmat(1 / R[i, :, n], N, 1) * Xinput[i].T.conjugate()) @ Xinput[i] / J
#                 w = LA.solve(W[:, :, i] @ D, e_n)
#                 W[n, :, i] = (w / np.sqrt(w.T.conjugate() @ D @ w)).T.conjugate()
#
#             for i in range(I):
#                 Y[i, :, n] = W[n, :, i].T.conjugate() @ Xinput[i, :, :].T
#
#         P = np.abs(Y) ** 2
#
#         cost = cost_function(P, R, W, I, J)
#         if draw:
#             costs[it, 0] = cost
#
#         print(it, cost)
#
#         """ projection back """
#         Z, D = projection_back(Y, X[:, :, refMic])
#
#         Y = Z
#         for i in range(I):
#             W[:, :, i] = D[:, :, i] @ W[:, :, i]  # update W
#
#         if not it % dump_step:
#             e.eval_loss_func(np.abs(Y))
#             # e.dump_wav(Y, it)
#             e.eval_mir(Y)
#
#         assert Xinput.shape == (I, J, M)
#
#     e.dump_sdrplot()
#     print('Done')
#     return Y, e


def bss_idlma(mix, models, song_num, output_path, ns, sig1, sig2, refMic, semi_supervised, dump_spectrogram, **params):
    """
    % [inputs]
    %        mix: observed mixture (len x mic)
    %         ns: number of sources (scalar)
    %         it: number of iterations (scalar)
    %       draw: plot cost function values or not (logic, true or false)
    semi_supervised: predict type

            sig1: reference sources 1 (for debug)
            sig2: reference sources 2 (for debug)
    """

    # Short-time Fourier transform
    fftSize = models[0].info.get('fftSize', 1024)
    shiftSize = fftSize // 2
    X, window = stft(mix, fftSize, shiftSize)

    S1, _ = stft(sig1[:, refMic], fftSize, shiftSize) # for debug
    S2, _ = stft(sig2[:, refMic], fftSize, shiftSize) # for debug
    e = Evaluate([sig1, sig2], [S1, S2], song_num, output_path, shiftSize=shiftSize)

    # Whitening (applying PCA)
    # Xwhite = whitening(X, ns)  # decorrelate input multichannel signal

    # IDLMA
    if semi_supervised:
        Y, e = idlma_semi_supervised(X, models, randomly_initW=True, refMic=refMic, e=e, dump_spectrogram=dump_spectrogram, **params)
    else:
        Y, e = idlma(X, models, randomly_initW=True, refMic=refMic, e=e, dump_spectrogram=dump_spectrogram, **params)

    if dump_spectrogram:
        for i in range(Y.shape[2]):
            spectrogram(np.abs(Y[:, :, i]), os.path.join('../dnn/', output_path, '{}_{}_idlma.png'.format(song_num, i)))

    sep = istft(Y, shiftSize, window, mix.shape[0]).T

    return sep, e
