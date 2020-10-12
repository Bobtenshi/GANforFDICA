import numpy as np
import os
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
import chainer.functions as F
from chainer import Variable
from collections import defaultdict
from stft import stft, istft
from dnn.divergence import itakura_saito_divergence
from stft import spectrogram


def v(x):
    return Variable(np.asarray(x, dtype=np.float32))


def format_result(res, print_SIRSAR=True, print_perm=True):
    if print_SIRSAR:
        print('---------------')
        print('SDR: {}'.format(' '.join(map(str, res[0]))))
        print('SIR: {}'.format(' '.join(map(str, res[1]))))
        print('SAR: {}'.format(' '.join(map(str, res[2]))))
    else:
        print('SDR: {}'.format(' '.join(map(str, res[0]))))

    if print_perm:
        print('perm: {}'.format(' '.join(map(str, res[3]))))
    return {
        'SDR': list(map(float, res[0])),
        'SIR': list(map(float, res[1])),
        'SAR': list(map(float, res[2])),
        'perm': list(map(float, res[3])),
    }

na = np.array


class Evaluate(object):

    def __init__(self, sig_list, S_list, song_num, output_path, shiftSize=512):
        """

        Args:
            sig_list: signal
            S_list: stft(signal)
        """
        self.sig_list = sig_list
        self.S_list = S_list
        self.sxr = []  # list(float)
        self.sdr = []  # list(dict)
        self.shiftSize = shiftSize
        self.song_num = song_num
        self.output_path = output_path
        self.res = defaultdict(list)

    def eval_loss_func(self, X, it=0):
        """
        (1-iterationごとの想定で) Xと正解のMSEを測る
        """
        print(it)
        losses = np.array(
            [itakura_saito_divergence(v(X[:, :, i]), v(np.abs(self.S_list[i]))).data for i in range(X.shape[2])]
        )
        print('IS loss : {}'.format(losses))
        losses = np.array(
            [F.mean_squared_error(v(X[:, :, i]), v(np.abs(self.S_list[i]))).data for i in range(X.shape[2])]
        )
        print('MSE loss : {}'.format(losses))

    def eval_mir(self, X):
        """
        (1-iterationごとの想定で) Xと正解のMSEを測る
        """
        signals = istft(X, length=len(self.sig_list[0]), shiftSize=self.shiftSize)
        res = bss_eval_sources(
            reference_sources=na([self.sig_list[0][:, 0], self.sig_list[1][:, 0]]),
            estimated_sources=signals.T
        )
        sdr_avg = res[0].mean()
        res = format_result(res, False, False)
        # self.sdr.append(sdr_avg)
        # self.sxr.append(res)
        self.report(**res)

    def report(self, **kwargs):
        """
        Args:
            **kwargs:
                report(hoge=5)
                hogeに5をレポートrepo-to
        Returns:
        """

        for key, val in kwargs.items():
            self.res[key].append(val)

    def dump_sdrplot(self, filepath='./sdr.png'):
        import matplotlib.pyplot as plt
        plt.plot(self.sdr)
        plt.savefig(filepath)

    def dump_spectrogram(self, X=None, suffix=''):
        """

        Args:
            X:
            suffix: iterationなど

        Returns:

        """
        if X is None:
            for i in range(len(self.S_list)):
                spectrogram(
                    np.abs(self.S_list[i]),
                    os.path.join('../dnn/', self.output_path, '{}_{}_ref.png'.format(self.song_num, i)),
                    vmin=-5
                )
        else:
            for i in range(X.shape[2]):
                spectrogram(
                    np.abs(X[:, :, i]),
                    os.path.join('../dnn/', self.output_path, '{}_{}_{}.png'.format(self.song_num, i, suffix)),
                    vmin=-5
                )
            print(os.path.join('../dnn/', self.output_path, '{}_{}_{}.png'.format(self.song_num, i, suffix)))

    def dump_wav(self, Xinput, suffix=''):
        signals = istft(Xinput, length=len(self.sig_list[0]), shiftSize=self.shiftSize)
        for i in range(signals.shape[1]):
            wavfile.write('./output/_out_{}_{}.wav'.format(i, suffix), 16000, signals[:, i])
