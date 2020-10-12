import os
import argparse
from datetime import datetime as dt
import numpy as np
import numpy.linalg as LA
from stft import stft, istft
from scipy.io import wavfile
from numpy.random import rand
from separation.projection_back import projection_back
from dnn.model import predict as mpredict
from separation.evaluater import Evaluate
from separation.common import read_wavs, load_models
import io_handler as IO
import notify


def local_covariance_init(X, N, K):
    # TODO implement
    I, J, M = X.shape
    R = np.random.rand(M, M, I, N)
    return R

def _det_2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def cost_function(X, R_x, invR_x, I, J):
    cost = 0
    # for i in range(I):
    #     for j in range(J):
    #         cost = cost + np.log(np.real(LA.det(np.pi * R_x[:, :, i, j])))

    if R_x.shape[0] == 2:
        cost += np.sum([[np.log(np.real(_det_2(np.pi * R_x[:, :, i, j]))) for i in range(I)] for j in range(J)])
    else:
        cost += np.sum([[np.log(np.real(LA.det(np.pi * R_x[:, :, i, j]))) for i in range(I)] for j in range(J)])
    XX = np.einsum('ijp,ijq->ijpq', X, X.conj())
    cost += np.real(np.einsum('ijpq,ijqp->', invR_x, XX))

    return cost

def duong_readable(X, N=2, n_iter=100, v=None, R=None):
    I, J, M = X.shape
    print('Initializing spatial covariance...')

    if v is None:
        v = np.ones([I, J, N], dtype=np.complex128)
        # R = local_covariance_init(X, N, 10*N)

    if R is None:
        # Identity Matrix
        R = np.zeros([M, M, I, N], dtype=np.complex128)
        for i in range(I):
            for n in range(N):
                R[:, :, i, n] = np.eye(M)

    chat = np.zeros([M, I, J, N], dtype=np.complex128)
    Rhat_c = np.zeros([M, M, I, J, N], dtype=np.complex128)
    W = np.zeros([M, M, I, J, N], dtype=np.complex128)

    R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4) M * M * I * J * N
    R_x = R_c.sum(axis=4)  # Eq.(5)

    now = dt.now()

    # Id = np.einsum('pq,ijn->pqijn', np.eye(M), np.ones([I, J, N]))
    invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])
    print(cost_function(X, R_x, invR_x, I, J))

    for it in range(n_iter):
        for i in range(I):
            for n in range(N):
                # E-step
                for j in range(J):
                    W[:, :, i, j, n] = R_c[:, :, i, j, n] @ LA.inv(R_x[:, :, i, j])  # Eq.(32)
                    chat[:, i, j, n] = W[:, :, i, j, n] @ X[i, j, :]  # Eq.(33)
                    Rhat_c[:, :, i, j, n] = np.einsum('i,j->ij', chat[:, i, j, n], chat[:, i, j, n].conjugate()) \
                                      + (np.eye(M) - W[:, :, i, j, n]) @ R_c[:, :, i, j, n]  # Eq.(34)

                # M-step
                invR = LA.inv(R[:, :, i, n])
                wRhat_c = np.zeros([M, M], dtype=np.complex128)
                for j in range(J):
                    v[i, j, n] = (1/M) * np.real(np.trace(invR @ Rhat_c[:, :, i, j, n])) # Eq.(35)
                    v[i, j, n] = max(v[i, j, n], 1e-10)
                    wRhat_c += Rhat_c[:, :, i, j, n] / v[i, j, n]  # Eq.(36)

                R[:, :, i, n] = (1/J) * wRhat_c  # Eq.(36)

        # covariance calculation
        R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4)
        R_x = R_c.sum(axis=4)  # Eq.(5)

        invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])

        print(it, cost_function(X, R_x, invR_x,  I, J))
        print('time: ', dt.now() - now)
        now = dt.now()

    print('Duong method with full-rank spatial covariance model done.')
    return v, R, Rhat_c


def duong_method(X, N=2, n_iter=100, v=None, R=None, evaluater=None):
    I, J, M = X.shape
    assert n_iter > 0

    if v is None:
        v = np.ones([I, J, N], dtype=np.complex128)
        # R = local_covariance_init(X, N, 10*N)

    if R is None:
        print('Initializing spatial covariance...')
        R = np.einsum('pq,in->pqin', np.eye(M), np.ones([I, N]), dtype=np.complex128)

    R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4) M * M * I * J * N
    R_x = R_c.sum(axis=4)  # Eq.(5)

    now = dt.now()

    Id = np.einsum('pq,ijn->pqijn', np.eye(M), np.ones([I, J, N]))
    invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])
    # print(cost_function(X, R_x, invR_x, I, J))

    for it in range(n_iter):
        invR = np.array([[LA.inv(R[:, :, i, n]) for n in range(N)] for i in range(I)])
        W = np.einsum('pqijn,ijqr->prijn', R_c, invR_x)
        chat = np.einsum('pqijn,ijq->pijn', W, X)
        Rhat_c = np.einsum('pijn,qijn->pqijn', chat, chat.conj()) + np.einsum('pqijn,qrijn->prijn', (Id - W), R_c)

        # Rとvの更新どっちを先にするか
        v = np.real(np.einsum('inpq,qpijn->ijn', invR, Rhat_c)) / M
        v = np.maximum(v, 1e-10)

        R = np.einsum('ijn,pqijn->pqin', 1/v, Rhat_c) / J

        # covariance calculation
        R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4)
        R_x = R_c.sum(axis=4)  # Eq.(5)

        invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])

        print(it, cost_function(X, R_x, invR_x,  I, J))
        print('time: ', dt.now() - now)
        now = dt.now()

        if evaluater is not None:
            # todo 整理
            evaluater.eval_mir(chat[0])

    print('Duong method with full-rank spatial covariance model done.')
    return chat, v, R


def duong_wrapper(mix, ns, fftSize, shiftSize, n_iter, fs, d, c, refMic):
    X, window = stft(mix, fftSize, shiftSize)
    # I, J, M = X.shape

    chat, v, R = duong_method(X, ns, n_iter)

    # TODO implement permutation_solver here

    # Y = np.zeros([M, I, J, ns], dtype=np.complex128)
    # for i in range(I):
    #     for j in range(J):
    #         vr = np.zeros([M, M], dtype=np.complex128)
    #         for n in range(ns):
    #             vr += v[i, j, n] * R[:, :, i, n]
    #         for n in range(ns):
    #             Y[:, i, j, n] = v[i, j, n] * R[:, :, i, n] @ LA.inv(vr) @ X[i, j, :]

    sep = istft(chat[refMic], shiftSize, window, len(mix))
    return sep


class DuongDNN(object):
    """
    todo
    I*A系でいうR(分散)とDuongのR(空間相関行列)がconflictして意味不明，可読性・・・
    """

    def __init__(self,
                 models,  # list of model
                 refMic=0,  # reference mic index
                 dump_step=10,  # 何回に1回Evaluaterを通すか
                 dump_spectrogram=False,
                 output_path='./output',
                 sampling_rate=8000,
                 song_num=0,  # song number of name (for output)
                 verbose=False,

                 # IDLMA parameters
                 n_iter=100,
                 delta=0.1,
                 delta_type='max',
                 n_update_w=10,
                 wf_type=0,
                 **params
    ):
        self.models = models
        self.refMic = refMic
        self.dump_step = dump_step
        self.dump_spectrogram = dump_spectrogram
        self.output_path = '../dnn/{}/'.format(output_path)
        self.sampling_rate = sampling_rate
        self.song_num = song_num
        self.verbose = verbose

        self.n_iter = n_iter
        self.delta = delta
        self.delta_type = delta_type
        self.wf_type = wf_type
        self.n_update_w = n_update_w

        self.evaluater = None
        self.Xinput = None
        self.R = None
        self.Rspace = None
        self.Y = None
        self.Yabs = None
        self.W = None
        self.I, self.J, self.M, self.N = None, None, None, None

    def init_param(self, random_init=False):
        """
        Initialize parameter R, Y, and W

        Args:
            randomly_initW:

        Returns:

        """
        Xinput = self.Xinput
        I, J, M = Xinput.shape
        N = len(self.models)
        self.I, self.J, self.M, self.N = I, J, M, N

        # Initialization
        self.R = rand(I, J, N)
        self.Y = np.zeros([I, J, N], dtype=np.complex128)

        for n in range(N):
            self.Y[:, :, n] = Xinput[:, :, self.refMic]

        self.Yabs = np.abs(self.Y)

        if random_init:
            # 複素正規分布の逆行列から生成したrank-1行列
            A = np.array([LA.inv(np.random.randn(N, M) * np.exp(np.random.rand(N, M) * 2 * np.pi * 1j)) for _ in range(I)])
            self.Rspace = np.einsum('inp,inq->pqin', A, A) + np.einsum('pq,in->pqin', np.eye(M), np.ones([I, N]), dtype=np.complex128) * 1e-8
        else:
            # identity matrix
            self.Rspace = np.einsum('pq,in->pqin', np.eye(M), np.ones([I, N]), dtype=np.complex128)

    def update_r(self):
        """ Update R """
        """
        todo M, N > 2に対応
        """
        Y, N, models, delta, delta_type, wf_type = self.Yabs, self.N, self.models, self.delta, self.delta_type, self.wf_type

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

    def duong_method(self, n_iter=100, v=None, R=None):
        I, J, M, N = self.I, self.J, self.M, self.N
        X = self.Xinput

        assert n_iter > 0

        if v is None:
            v = np.ones([I, J, N], dtype=np.complex128)
            # R = local_covariance_init(X, N, 10*N)

        if R is None:
            print('Initializing spatial covariance...')
            R = np.einsum('pq,in->pqin', np.eye(M), np.ones([I, N]), dtype=np.complex128)

        R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4) M * M * I * J * N
        R_x = R_c.sum(axis=4)  # Eq.(5)

        now = dt.now()

        Id = np.einsum('pq,ijn->pqijn', np.eye(M), np.ones([I, J, N]))
        invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])
        invR = np.array([[LA.inv(R[:, :, i, n]) for n in range(N)] for i in range(I)])
        W = np.einsum('pqijn,ijqr->prijn', R_c, invR_x)
        chat = np.einsum('pqijn,ijq->pijn', W, X)

        print(cost_function(X, R_x, invR_x, I, J))

        for it in range(n_iter):

            Rhat_c = np.einsum('pijn,qijn->pqijn', chat, chat.conj()) + np.einsum('pqijn,qrijn->prijn', (Id - W), R_c)

            # vの更新はしない
            # v = np.real(np.einsum('inpq,qpijn->ijn', invR, Rhat_c)) / M
            # v = np.maximum(v, 1e-10)

            R = np.einsum('ijn,pqijn->pqin', 1/v, Rhat_c) / J

            # covariance calculation
            R_c = np.einsum('ijn,pqin->pqijn', v, R)  # Eq.(4)
            R_x = R_c.sum(axis=4)  # Eq.(5)

            invR_x = np.array([[LA.inv(R_x[:, :, i, j]) for j in range(J)] for i in range(I)])
            invR = np.array([[LA.inv(R[:, :, i, n]) for n in range(N)] for i in range(I)])
            W = np.einsum('pqijn,ijqr->prijn', R_c, invR_x)
            chat = np.einsum('pqijn,ijq->pijn', W, X)

            cost = cost_function(X, R_x, invR_x,  I, J)
            print(it, cost)
            # print('time: ', dt.now() - now)
            # now = dt.now()

            self.evaluater.eval_mir(chat[0])
            self.evaluater.report(cost=cost)

        print('Duong method with full-rank spatial covariance model done.')

        # Arie論文ではこれをvではなくzと表記している
        # 次のDNNに渡す用
        z = np.real(np.einsum('inpq,qpijn->ijn', invR, Rhat_c)) / M
        z = np.maximum(z, 1e-10)

        return chat, z, R


    def multichannel_wiener_filter(self, v, R):
        """
        estimate signal using Multi WF
        """
        M, I, J, X, N = self.M, self.I, self.J, self.Xinput, self.N

        Y = np.zeros([M, I, J, N])
        for i in range(I):
            for j in range(J):
                vr = np.zeros([M, M])
                for n in range(N):
                    vr += v[i, j, n] * R[:, :, i, n]
                for n in range(N):
                    Y[:, i, j, n] = v[i, j, n] * R[:, :, i, n] * LA.inv(vr) * X[i, j, :]

        self.Y = Y

    def set_evaluater(self, signals):
        fftSize = self.models[0].info.get('fftSize')
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
            self.evaluater.dump_spectrogram(self.Y, '{}_duongDNN'.format(it))

    def iterate(self):
        self.init_param()
        self.report_eval()

        for it in range(1, self.n_iter+1):
            # if not (it - 1) % self.n_update_w:
            self.update_r()
                # if it > 1:
                #     self.show_sn()

            # IDLMAでのRはduong法でのvに対応する
            chat, v, Rspace = self.duong_method(n_iter=self.n_update_w, v=self.R, R=self.Rspace)

            self.Rspace = Rspace
            self.Yabs = np.sqrt(v)
            self.Y = chat[self.refMic]

            # if self.verbose:
            #     cost = cost_function(P, self.R, self.W, self.I, self.J)
            #     print(it, cost)

            # if not it % self.dump_step:
            # self.report_eval(it)
            self.evaluater.dump_spectrogram(self.Y, '{}_duongDNN'.format(it * self.n_update_w))

    def dump_wav(self, mix, sep):
        # --- write wav ---
        wavfile.write(os.path.join(self.output_path, 'mix_{}.wav'.format(self.song_num)), self.sampling_rate, mix)
        for j, m in enumerate(self.models):
            f = m.info['target'][:-4]
            fname = os.path.join(self.output_path, '{}_{}_duongdnn.wav'.format(f, self.song_num))
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

        # evaluate par iteration step ( or dump_step )
        self.set_evaluater(signals)

        # run!
        self.iterate()

        sep = istft(self.Y, shiftSize, window, mix.shape[0]).T
        self.dump_wav(mix, sep)

        return sep


def test():
    refMic = 0
    fsResample = 8000
    ns = 2
    fftSize = 1024
    shiftSize = 512
    micSpacing = 0.0556
    c = 344.82
    it = 5

    if False:
        signals = read_wavs(301, 1, ['drums', 'vocals'], 8000)
    else:
        fs, sig1 = wavfile.read('./input/guitar.wav')
        fs, sig2 = wavfile.read('./input/synth.wav')
        assert fs == 8000
        sig1 = sig1 / 32768
        sig2 = sig2 / 32768
        signals = [sig1, sig2]

    mix = np.array(signals).sum(axis=0)

    sep = duong_wrapper(mix, ns, fftSize, shiftSize, it, fsResample, micSpacing, c, refMic)

    from mir_eval.separation import bss_eval_sources
    res_org = bss_eval_sources(
        reference_sources=np.array([signals[0][:, 0], signals[1][:, 0]]),
        estimated_sources=np.array([mix[:, 0], mix[:, 0]])
    )
    res = bss_eval_sources(
        reference_sources=np.array([signals[0][:, 0], signals[1][:, 0]]),
        estimated_sources=sep.T
    )
    from separation.evaluater import format_result
    format_result(res_org)
    format_result(res)


if __name__ == '__main__':
    # exit(test())
    p = argparse.ArgumentParser()
    p.add_argument('-song', '--song_num', metavar='N', type=int, default=0,
                   help='song number in DSD100')
    p.add_argument('-m', '--model_path', metavar='N', type=str,
                   help='model_path( ex output -> ../dnn/outputd/model.npz')

    p.add_argument('-sr', '--sampling_rate', metavar='N', type=int, default=8000,
                   help='sampling_rate [Hz]')
    p.add_argument('-i', '--n_iter', metavar='N', type=int, default=10,
                   help='number of iterations (default: 100)')
    p.add_argument('-r', '--refMic', metavar='N', type=int, default=0,
                   help='refMic (default: 0)')
    p.add_argument('-d', '--dump_step', metavar='N', type=int, default=10,
                   help='dump wav and loss for each n epoch (default: 10)')
    p.add_argument('-dl', '--delta', metavar='N', type=float, default=0.1,
                   help='power + delta (default: 0.01)')  # 分散のバイアス項
    p.add_argument('-dt', '--delta_type', metavar='N', type=str, default='max',
                   help='delta type (default: max)')
    p.add_argument('-w', '--n_update_w', metavar='N', type=int, default=10,
                   help='how many updates w per dnn predict (default: 10)')  # DNN1回ごとに何回duongのEMを通すか
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
    duong = DuongDNN(models, output_path=args.model_path, **args.__dict__)

    if args.song_num > 0:
        signals = read_wavs(args.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
        duong.main(signals)
    else:
        # 評価用の25曲(曲長30秒)
        res = {}

        for song_num in range(301, 301 + 25):
            duong.song_num = song_num
            signals = read_wavs(duong.song_num, args.space_type, args.files.split('/'), args.sampling_rate)
            duong.main(signals)
            res[song_num] = dict(duong.evaluater.res)

        print(res)
        IO.write_json(os.path.join('./json', 'sdr_imp_DuongDNN_{}{}.json'.format(args.model_path, args.suffix)), res)

        sdrimps = [round(np.mean(res[301 + i]['SDR'][-1]), 2) for i in range(25)]
        import notify
        notify.main(
            '\n'.join([
            'DuongDNN_{}{}'.format(args.model_path, args.suffix),
            '/'.join(map(str, sdrimps)),
            str(np.mean(sdrimps)),
            ])
        )