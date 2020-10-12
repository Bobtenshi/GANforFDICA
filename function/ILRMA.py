import numpy as np
import numpy.linalg as LA
from numpy.random import rand

import os
import glob
import argparse
from scipy.io import wavfile

import io_handler as IO
from stft import stft, istft
from separation.common import cost_function, read_wavs
from separation.evaluater import Evaluate
from separation.projection_back import projection_back

na = np.asarray


class ILRMA(object):
    def __init__(self,
                 files, # ex) bass/vocals
                 refMic=0,  # reference mic index
                 dump_step=10,  # 何回に1回Evaluaterを通すか
                 dump_spectrogram=False,
                 output_path='./output',
                 sampling_rate=8000,
                 song_num=0,  # song number of name (for output)
                 verbose=False,

                 # ILRMA parameters
                 n_iter=100,
                 nb=20, # number of bases
                 fftSize=4096,
                 randomly_initW=False,
                 **params
    ):
        self.method = 'ILRMA'
        self.files = files

        self.refMic = refMic
        self.dump_step = dump_step
        self.dump_spectrogram = dump_spectrogram
        self.output_path = output_path
        self.sampling_rate = sampling_rate
        self.song_num = song_num
        self.verbose = verbose

        self.n_iter = n_iter
        self.randomly_initW = randomly_initW
        self.fftSize = fftSize
        self.nb = nb

        self.evaluater = None

        self.Xinput = None
        self.R = None
        self.T, self.V = None, None
        self.Y = None
        self.W = None
        self.I, self.J, self.M, self.N = None, None, None, None

    def init_param(self):
        """
        Initialize parameter R, Y, and W

        Args:
            randomly_initW: Wの初期値をランダムにするか単位行列にするか

        Returns:

        """
        Xinput = self.Xinput
        I, J, M = Xinput.shape
        N = 2 # todo hard coding
        self.I, self.J, self.M, self.N = I, J, M, N

        self.W = np.zeros([N, M, I], dtype=np.complex128)
        if self.randomly_initW:
            # 複素正規分布
            for i in range(I):
                self.W[:, :, i] = np.random.randn(N, M) * np.exp(np.random.rand(N, M) * 2 * np.pi * 1j)
        else:
            for i in range(I):
                self.W[:, :, i] = np.eye(N)

        # Initialization
        self.R = rand(I, J, N)
        self.Y = np.zeros([I, J, N], dtype=np.complex128)

        # spectrogram(np.abs(Xinput[:, :, refMic]), output_path='spec_org.png')
        for n in range(N):
            self.Y[:, :, n] = Xinput[:, :, self.refMic]

        L = self.nb
        self.T = rand(I, L, N)
        self.V = rand(L, J, N)

    def update_r(self):
        """ Update R """
        """
        todo M, N > 2に対応
        R_0 -> DNNでupdate
        R_1 -> NMFでupdate
        """
        Y, N = self.Y, self.N
        P = np.abs(Y) ** 2

        for n in range(N):
            T, V, R = self.T, self.V, self.R
            """ Update T """
            self.T[:,:,n] = T[:,:,n] * ( np.sqrt( (P[:,:,n] * (R[:,:,n] ** (-2))) @ V[:,:,n].T / ( (R[:,:,n] ** (-1)) @ V[:,:,n].T ) ) )
            self.T[:,:,n] = np.maximum(T[:,:,n], 1e-15)
            self.R[:,:,n] = T[:,:,n] @ V[:,:,n]

            T, V, R = self.T, self.V, self.R
            """ Update V """
            self.V[:,:,n] = V[:,:,n] * ( np.sqrt( T[:,:,n].T @ (P[:,:,n] * (R[:,:,n] ** (-2))) / ( T[:,:,n].T @ (R[:,:,n] ** (-1)) ) ) )
            self.V[:,:,n] = np.maximum(V[:,:,n], 1e-15)
            self.R[:,:,n] = T[:,:,n] @ V[:,:,n]

    def update_w(self):
        """
        update separation Matrix W and predicted spectrogram Y
        """
        Xinput, R, N, M, I, J = self.Xinput, self.R, self.N, self.M, self.I, self.J
        U = np.zeros([I, N, M, M], dtype=np.complex128)

        for n in range(N):
            """ Update W """
            e_n = np.array([1 if _n == n else 0 for _n in range(N)])
            for i in range(I):
                U[i, n] = (np.matlib.repmat(1 / R[i, :, n], N, 1) * Xinput[i].T.conjugate()) @ Xinput[i] / J
                w = LA.solve(self.W[:, :, i] @ U[i, n], e_n)
                self.W[n, :, i] = (w / np.sqrt(w.T.conjugate() @ U[i, n] @ w)).T.conjugate()

        for n in range(N):
            for i in range(I):
                self.Y[i, :, n] = self.W[n, :, i].T.conjugate() @ Xinput[i, :, :].T

    def set_evaluater(self, signals):
        fftSize = self.fftSize
        shiftSize = fftSize // 2

        S = []
        for signal in signals:
            _S, _ = stft(signal[:, self.refMic], fftSize, shiftSize)
            S.append(_S)

        self.evaluater = Evaluate(signals, S, self.song_num, self.output_path, shiftSize=shiftSize)

    def report_eval(self, it=0):
        self.evaluater.eval_loss_func(np.abs(self.Y), it)
        self.evaluater.eval_mir(self.Y)
        if self.dump_spectrogram:
            self.evaluater.dump_spectrogram(self.Y, '{}_{}'.format(it, self.method))

    def projection_back(self):
        """ projection back """
        Xinput, I = self.Xinput, self.I
        Z, D = projection_back(self.Y, Xinput[:, :, self.refMic])

        self.Y = Z
        for i in range(I):
            self.W[:, :, i] = D[:, :, i] @ self.W[:, :, i]  # update W

    def iterate(self):
        self.init_param()
        self.report_eval()

        for it in range(1, self.n_iter+1):
            self.update_r()
            self.update_w()
            self.projection_back()

            if self.verbose:
                P = np.abs(self.Y) ** 2
                cost = cost_function(P, self.R, self.W, self.I, self.J)
                print(it, cost)

            if not it % self.dump_step:
                self.report_eval(it)

    def theoritical_solution(self):
        S = np.array(self.evaluater.S_list)
        X = self.Xinput

        for i in range(self.I):
            # A = X[i].T @ S[:, i, :].T.conj() @ LA.inv(S[:, i, :] @ S[:, i, :].T.conj())
            # self.W[:, :, i] = LA.inv(A)
            self.W[:, :, i] = S[:, i, :] @ X[i].conj() @ LA.inv(X[i].T @ X[i].conj())

            self.Y[i] = X[i] @ self.W[:, :, i].T

        # print(np.abs(S).mean(), np.abs(self.Y).mean())

        self.report_eval()

    def dump_wav(self, mix, sep):
        # --- write wav ---
        if self.output_path is None: return
        wavfile.write(os.path.join(self.output_path, 'mix_{}.wav'.format(self.song_num)), self.sampling_rate, mix)
        for j, f in enumerate(self.files.split('/')):
            fname = os.path.join(self.output_path, '{}_{}_{}.wav'.format(f, self.song_num, self.method))
            print('->', fname)
            wavfile.write(fname, self.sampling_rate, sep[j])
        print('write complete')

    def main(self, signals, **params):
        """
        パラメータはIDLMAのインスタンスクラスが持っていて，混合音を入力として分離音が出力される
        性能評価はEvaluaterに託す
        """
        nch = len(signals)
        mix = np.array(signals).sum(axis=0)

        # Short-time Fourier transform
        fftSize = self.fftSize
        shiftSize = fftSize // 2
        X, window = stft(mix, fftSize, shiftSize)
        self.Xinput = X

        # evaluate per iteration step ( or dump_step )
        self.set_evaluater(signals)

        # run!
        if self.n_iter:
            self.iterate()
        else:
            # 理想解を求めて終了
            self.init_param()
            self.theoritical_solution()

        sep = istft(self.Y, shiftSize, window, mix.shape[0]).T
        self.dump_wav(mix, sep)

        return sep


if __name__ == '__main__':
    """
    ex)
    $ python ILRMA.py -f 4096 -fls vocals/bass
    """

    p = argparse.ArgumentParser()
    p.add_argument('-song', '--song_num', metavar='N', type=int, default=0,
                   help='song number in DSD100')
    p.add_argument('-n', '--nb', metavar='N', type=int, default=20,
                   help='number of bases (default: 20)')
    p.add_argument('-sr', '--sampling_rate', metavar='N', type=int, default=8000,
                   help='sampling_rate [Hz]')
    p.add_argument('-i', '--n_iter', metavar='N', type=int, default=100,
                   help='number of iterations (default: 100) 0にすると最小二乗解を計算してSDRを算出する')
    p.add_argument('-r', '--refMic', metavar='N', type=int, default=0,
                   help='refMic (default: 0)')
    p.add_argument('-d', '--dump_step', metavar='N', type=int, default=10,
                   help='dump wav and loss for each n epoch (default: 10)')
    p.add_argument('-s', '--dump_spectrogram', action='store_true', default=False,
                   help='dump spectrogram png')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='dump cost function value per iteration step')
    p.add_argument('-sp', '--space_type', metavar='N', type=int, default=1,
                   help='収録環境 1->IR1 2->IR2')
    p.add_argument('-sf', '--suffix', metavar='N', default='', type=str,
                   help='suffix of output json name sdr_imp_ILRMA_{suffix}.json ')
    p.add_argument('-fls', '--files', metavar='N', type=str, default='',
                   help='use instruments ex) vocals/bass')
    p.add_argument('-o', '--output_path', metavar='N', type=str, default='./output',
                   help='output_path')
    p.add_argument('-f', '--fftSize', metavar='N', type=int, default=4096,
                   help=' window length in STFT (default: 4096)')
    args = p.parse_args()
    args.model_path = ''

    ilrma = ILRMA(**args.__dict__)

    if args.song_num > 0:
        signals = read_wavs(args.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
        sdr_imp = ilrma.main(signals)
    else:
        # 評価用の25曲(曲長30秒)
        res = {}

        for song_num in range(301, 301 + 25):
            ilrma.song_num = song_num
            signals = read_wavs(ilrma.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
            ilrma.main(signals)
            res[song_num] = dict(ilrma.evaluater.res)

        print(res)
        IO.write_json(os.path.join('./json', 'sdr_imp_ILRMA{}.json'.format(args.suffix)), res)

        import notify
        notify.sdr_imp(res, args, method='ILRMA')