import glob, os
import numpy as np
import numpy.linalg as LA
from dnn.model import load_model
from dnn.make_dataset2 import preprocess
from scipy.io import wavfile
from path import source_path


def cost_function(P, R, W, I, J):
    A = np.zeros([I, 1])
    for i in range(I):
        x = np.abs(LA.det(W[:, :, i]))
        x = max(x, 1e-10)
        A[i] = 2 * np.log(x)
    return -J * A.sum() + (P / R + np.log(R)).sum()


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


def song_num_to_file(song_num, target=None):
    """
    曲番号 -> ファイル名
    Args:
        song_num:
        target:

    Returns:

    """
    song_num = int(song_num)
    if song_num <= 50:
        path = os.path.join(source_path, 'Test/{:03d}*'.format(song_num))
    elif song_num <= 100:
        path = os.path.join(source_path, 'Dev/{:03d}*'.format(song_num))
    else:
        path = os.path.join(source_path, 'Addition/{:03d}*'.format(song_num))

    dirs = glob.glob(path)
    assert len(dirs) == 1, (path, dirs)

    p = dirs[0]

    # if target is None:
    #     wavs = []
    #     for f in files:
    #         wavs.append(os.path.join(p, f))
    #     return wavs

    mix_wav = os.path.join(p.replace('Sources', 'Mixtures'), 'mixture.wav')
    ref_wav = os.path.join(p, target)

    return mix_wav, ref_wav


def read_wavs(song_num, space_type, instruments, sampling_rate=8000):
    """
    Args:
        song_num: int
            001 ~ 050 Test
            051 ~ 100 Dev
            101 ~ 自作
                301 ~ 325 : 001 ~ 025(30s ~ 60s)
        space_type: int (1 or 2)
            1: pos 050, 130 mic 21, 23
            2: pos 050, 110 mic 21, 22
        instruments: list(str)
            ex) ['vocals', 'drums']
        sampling_rate: int

    Returns:
        list of signals
    """

    if space_type == 0:
        files = list(map(lambda x: x + '.wav', instruments))
    elif space_type == 1:
        files = list(map(lambda x: x + '_E2A_pos050130_mic2123.wav', instruments))
    elif space_type == 2:
        files = list(map(lambda x: x + '_E2A_pos050110_mic2122.wav', instruments))
    else:
        raise ValueError

    if sampling_rate == 8000:
        _8k = True
    elif sampling_rate == 16000:
        _8k = False
    else:
        # TODO sampling_rate other than 8k,16k are not supported
        raise NotImplementedError

    signals = []
    for file in files:
        _, ref_wav = song_num_to_file(song_num, file)
        print(ref_wav)
        fs, signal = wavfile.read(ref_wav)
        assert fs == 16000

        signal = preprocess(signal, _8k)
        signals.append(signal)

    # length of all signals should be equal
    len_signal = len(signals[0])
    assert all(map(lambda s: len(s) == len_signal, signals))

    return signals


def load_models(model_dir, instruments, model_file='model', gpu_id=0):
    """
    Args:
        model_dir: str
        instruments: list(str)
            ex) ['vocals', 'drums']
        model_file:
        gpu_id: int

    Returns:
        list of DNN models
    """
    models = []
    for inst in instruments:
        model = load_model('../dnn/{}/{}'.format(model_dir, inst[0]), model_name=model_file, gpu_id=gpu_id)
        models.append(model)
    return models