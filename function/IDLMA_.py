import numpy as np
import numpy.linalg as LA
from numpy.random import rand

import os
import glob
import argparse
from scipy.io import wavfile

import io_handler as IO
from stft import stft, istft
from dnn.model import predict as mpredict
from separation.common import cost_function, read_wavs, load_models
from separation.evaluater import Evaluate
from separation.projection_back import projection_back

na = np.asarray


def modify_permutation(U, w, i):
    """
    バグがある可能性

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


class IDLMA(object):
    def __init__(self,
                 models,  # list of model
                 refMic=0,  # reference mic index
                 dump_step=10,  # 何回に1回Evaluaterを通すか
                 dump_spectorgram=False,
                 output_path='./output',
                 sampling_rate=8000,
                 song_num=0,  # song number of name (for output)
                 verbose=False,

                 # IDLMA parameters
                 n_iter=100,
                 delta=0.01,
                 delta_type='max',
                 n_update_w=10,
                 wf_type=0,
                 w_init='gauss',
                 **params
    ):
        self.models = models
        self.refMic = refMic
        self.dump_step = dump_step
        self.dump_spectrogram = dump_spectorgram
        self.output_path = '../dnn/{}/'.format(output_path)
        self.sampling_rate = sampling_rate
        self.song_num = song_num
        self.verbose = verbose

        self.n_iter = n_iter
        self.delta = delta
        self.delta_type = delta_type
        self.wf_type = wf_type
        self.w_init = w_init
        self.n_update_w = n_update_w

        self.evaluater = None
        self.Xinput = None
        self.R = None
        self.Y = None
        self.W = None
        self.I, self.J, self.M, self.N = None, None, None, None

    def init_param(self):
        """
        Initialize parameter R, Y, and W

        """
        Xinput = self.Xinput
        I, J, M = Xinput.shape
        N = len(self.models)
        self.I, self.J, self.M, self.N = I, J, M, N

        self.W = np.zeros([N, M, I], dtype=np.complex128)
        if self.w_init == 'gauss':
            # 複素正規分布
            for i in range(I):
                self.W[:, :, i] = np.random.randn(N, M) * np.exp(np.random.rand(N, M) * 2 * np.pi * 1j)
        elif self.w_init == 'id':
            print('Winit : identity')
            for i in range(I):
                self.W[:, :, i] = np.eye(N)

        # Initialization
        self.R = rand(I, J, N)
        self.Y = np.zeros([I, J, N], dtype=np.complex128)

        # spectrogram(np.abs(Xinput[:, :, refMic]), output_path='spec_org.png')
        for n in range(N):
            self.Y[:, :, n] = Xinput[:, :, self.refMic]

    def show_sn(self):
        """
        SN比をevaluaterにreport
        """
        Y, models = self.Y, self.models
        s00 = mpredict(models[0], Y[:, :, 0]).data.T ** 2 + 1e-5
        s01 = mpredict(models[0], Y[:, :, 1]).data.T ** 2 + 1e-5
        s10 = mpredict(models[1], Y[:, :, 0]).data.T ** 2 + 1e-5
        s11 = mpredict(models[1], Y[:, :, 1]).data.T ** 2 + 1e-5
        sn0 = s00.sum() / s01.sum()
        sn1 = s11.sum() / s10.sum()

        print('Y1:{} \nY2:{}'.format(sn0, sn1))
        self.evaluater.report(SN=float(np.log(sn0 + sn1)))

    def update_r(self):
        """ Update R """
        """
        todo M, N > 2に対応
        """
        Y, N, models, delta, delta_type, wf_type = self.Y, self.N, self.models, self.delta, self.delta_type, self.wf_type

        if wf_type == 0:
            for n in range(N):
                self.R[:, :, n] = mpredict(models[n], Y[:, :, n]).data.T ** 2

        elif wf_type == 1:
            for n in range(N):
                pred0 = mpredict(models[0], Y[:, :, n]).data.T ** 2 + 1e-5
                pred1 = mpredict(models[1], Y[:, :, n]).data.T ** 2 + 1e-5

                if n == 0:
                    self.R[:, :, 0] = np.abs(Y[:, :, 0]) * (pred0 / (pred0 + pred1)) ** 2
                elif n == 1:
                    self.R[:, :, 1] = np.abs(Y[:, :, 1]) * (pred1 / (pred0 + pred1)) ** 2

        elif wf_type == 2:
            s00 = mpredict(models[0], Y[:, :, 0]).data.T ** 2 + 1e-5
            s01 = mpredict(models[0], Y[:, :, 1]).data.T ** 2 + 1e-5
            s10 = mpredict(models[1], Y[:, :, 0]).data.T ** 2 + 1e-5
            s11 = mpredict(models[1], Y[:, :, 1]).data.T ** 2 + 1e-5
            self.R[:, :, 0] = np.abs(Y[:, :, 0] * (s00 / (s00 + s10)) + Y[:, :, 1] * (s01 / (s01 + s11))) ** 2
            self.R[:, :, 1] = np.abs(Y[:, :, 0] * (s10 / (s00 + s10)) + Y[:, :, 1] * (s11 / (s01 + s11))) ** 2

        elif wf_type == 3:
            s00 = mpredict(models[0], Y[:, :, 0]).data.T ** 2 + 1e-5
            s01 = mpredict(models[0], Y[:, :, 1]).data.T ** 2 + 1e-5
            s10 = mpredict(models[1], Y[:, :, 0]).data.T ** 2 + 1e-5
            s11 = mpredict(models[1], Y[:, :, 1]).data.T ** 2 + 1e-5
            self.R[:, :, 0] = (np.abs(Y[:, :, 0] + Y[:, :, 1]) * ( (s00 + s01) / (s00 + s01 + s10 + s11) ) ) ** 2
            self.R[:, :, 1] = (np.abs(Y[:, :, 0] + Y[:, :, 1]) * ( (s10 + s11) / (s00 + s01 + s10 + s11) ) ) ** 2

        else:
            raise ValueError

        for n in range(N):
            _delta = self.R[:, :, n].mean() * delta
            if delta_type == 'add':
                self.R[:, :, n] = self.R[:, :, n] + _delta
            elif delta_type == 'max':
                self.R[:, :, n] = np.maximum(self.R[:, :, n], _delta)
            else:
                raise ValueError

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

        # if not (it - 10) % n_update_w:
        #     for i in range(I):
        #         W[:, :, i] = modify_permutation(U[i], W[:, :, i], i)

        for n in range(N):
            for i in range(I):
                self.Y[i, :, n] = self.W[n, :, i].T.conjugate() @ Xinput[i, :, :].T

    def update_w2(self):
        """
        update separation Matrix W and predicted spectrogram Y
        """
        Xinput, R, N, M, I, J = self.Xinput, self.R, self.N, self.M, self.I, self.J
        U = np.zeros([I, N, M, M], dtype=np.complex128)

        XXH = np.einsum('ijp,ijq->ijpq', Xinput.conj(), Xinput)

        for n in range(N):
            U[:, n] = np.einsum('ij,ijpq->ipq', 1/R[:, :, n], XXH) / J
            e_n = np.array([1 if _n == n else 0 for _n in range(N)])

            w = np.array([LA.solve(self.W[:, :, i] @ U[i, n], e_n) for i in range(I)])
            wUw = np.einsum('ip,ipq,iq->i', w.conj(), U[:, n], w)

            self.W[n, :, :] = np.einsum('ip,i->pi', w, 1 / np.sqrt(wUw)).conj()
            self.Y[:, :, n] = np.einsum('mi,ijm->ij', self.W[n, :, :].conj(), Xinput)


    def set_evaluater(self, signals, song_num=None):
        fftSize = self.models[0].info.get('fftSize')
        shiftSize = fftSize // 2

        S = []
        for signal in signals:
            _S, _ = stft(signal[:, self.refMic], fftSize, shiftSize)
            S.append(_S)

        self.evaluater = Evaluate(signals, S, song_num, self.output_path, shiftSize=shiftSize)

    def report_eval(self, it=0):
        self.evaluater.eval_loss_func(np.abs(self.Y), it)
        self.evaluater.eval_mir(self.Y)
        if self.dump_spectrogram:
            self.evaluater.dump_spectrogram()

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
            if not (it - 1) % self.n_update_w:
                self.update_r()
                # if it > 1:
                #     self.show_sn()

            self.update_w()
            self.projection_back()

            if self.verbose:
                P = np.abs(self.Y) ** 2
                cost = cost_function(P, self.R, self.W, self.I, self.J)
                print(it, cost)

            if not it % self.dump_step:
                self.report_eval(it)

    def dump_wav(self, mix, sep):
        # --- write wav ---
        wavfile.write(os.path.join(self.output_path, 'mix_{}.wav'.format(self.song_num)), self.sampling_rate, mix)
        for j, m in enumerate(self.models):
            f = m.info['target'][:-4]
            fname = os.path.join(self.output_path, '{}_{}_idlma.wav'.format(f, self.song_num))
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
        fftSize = self.models[0].info.get('fftSize')
        shiftSize = fftSize // 2
        X, window = stft(mix, fftSize, shiftSize)
        self.Xinput = X

        # evaluate per iteration step ( or dump_step )
        self.set_evaluater(signals)

        # run!
        self.iterate()

        sep = istft(self.Y, shiftSize, window, mix.shape[0]).T
        self.dump_wav(mix, sep)

        return sep


if __name__ == '__main__':
    """
    ex)
    $ python IDLMA.py -m output_test -fls vocals/bass
    """
    # dir_target_map = {
    #     'd': 'drums.wav', 'v': 'vocals.wav',
    #     'b': 'bass.wav', 'o': 'other.wav',
    #     'g': 'guitar.wav', 's': 'synth.wav',
    # }

    p = argparse.ArgumentParser()
    p.add_argument('-song', '--song_num', metavar='N', type=int, default=0,
                   help='song number in DSD100')
    p.add_argument('-m', '--model_path', metavar='N', type=str,
                   help='model_path( ex output -> ../dnn/outputd/model.npz')

    p.add_argument('-sr', '--sampling_rate', metavar='N', type=int, default=8000,
                   help='sampling_rate [Hz]')
    p.add_argument('-i', '--n_iter', metavar='N', type=int, default=100,
                   help='number of iterations (default: 100)')
    p.add_argument('-r', '--refMic', metavar='N', type=int, default=0,
                   help='refMic (default: 0)')
    p.add_argument('-d', '--dump_step', metavar='N', type=int, default=10,
                   help='dump wav and loss for each n epoch (default: 10)')
    p.add_argument('-dl', '--delta', metavar='N', type=float, default=0.01,
                   help='power + delta (default: 0.01)')  # 分散のバイアス項
    p.add_argument('-dt', '--delta_type', metavar='N', type=str, default='max',
                   help='delta type (default: max)')
    p.add_argument('-winit', '--w_init', metavar='N', type=str, default='gauss',
                   help='separation matrix W initialization method id / gauss(default: gauss)')
    p.add_argument('-w', '--n_update_w', metavar='N', type=int, default=10,
                   help='how many updates w per dnn predict (default: 10)')  # DNN1回ごとに何回IPを通すか
    p.add_argument('-g', '--gpu_id', metavar='N', type=int, default=0,
                   help='gpu id (-1 if use cpu)')
    # p.add_argument('-semi', '--semi_supervised', action='store_true', default=False,
    #                help='semi_supervised_IDLMA')
    p.add_argument('-mf', '--model_file', metavar='N', default='model', type=str,
                   help='model_filename ex)model model_10')
    p.add_argument('-s', '--dump_spectrogram', action='store_true', default=False,
                   help='dump spectrogram png')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='dump cost function value per iteration step')
    p.add_argument('-wf', '--wf_type', metavar='N', type=int, default=0,
                   help='winner filter type (default: 0)')
    p.add_argument('-sp', '--space_type', metavar='N', type=int, default=1,
                   help='収録環境 1->IR1 2->IR2')
    p.add_argument('-sf', '--suffix', metavar='N', default='', type=str,
                   help='suffix of output json name sdr_imp_IDLMA_{output_file}{suffix}.json ')
    p.add_argument('-fls', '--files', metavar='N', type=str, default='',
                   help='use instruments ex) vocals/bass')
    args = p.parse_args()

    models = load_models(args.model_path, args.files.split('/'), args.model_file, args.gpu_id)
    args.output_path = args.model_path

    idlma = IDLMA(models, **args.__dict__)

    if args.song_num > 0:
        signals = read_wavs(args.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
        sdr_imp = idlma.main(signals)
    else:
        # 評価用の25曲(曲長30秒)
        # idlma.output_path = args.model_path
        res = {}

        for song_num in range(301, 301 + 25):
            idlma.song_num = song_num
            signals = read_wavs(idlma.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
            idlma.main(signals)
            res[song_num] = dict(idlma.evaluater.res)

        print(res)
        IO.write_json(os.path.join('./json', 'sdr_imp_IDLMA_{}{}.json'.format(args.model_path, args.suffix)), res)

        import notify
        sdrimps = [round(np.mean(res[301 + i]['SDR'][-1]), 3) for i in range(25)]
        notify.main('/'.join(map(str, sdrimps)))
        notify.main(np.mean(sdrimps))
