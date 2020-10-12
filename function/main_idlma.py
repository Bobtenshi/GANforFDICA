import os
import glob
import numpy as np
import argparse
from chainer import serializers
from scipy.io import wavfile
from mir_eval.separation import bss_eval_sources
from separation.bss_idlma import bss_idlma
from dnn.model import load_model
from dnn.make_datasets import preprocess, files
from dnn.evaluate import song_num_to_file
from dust.res import mean_sdr_imp
import notify
import io_handler as IO
from stft import spectrogram

# Setting parameters
# seed = 4  # seed of pseudo random values
refMic = 0  # reference mic for evaluation
_8k = True
if _8k:
    fsResample = 8000
else:
    fsResample = 16000  # resampling frequency

def format_result(res, print_perm=True):
    print('---------------')
    print('SDR: {}'.format(' '.join(map(str, res[0]))))
    print('SIR: {}'.format(' '.join(map(str, res[1]))))
    print('SAR: {}'.format(' '.join(map(str, res[2]))))
    if print_perm:
        print('perm: {}'.format(' '.join(map(str, res[3]))))

na = np.asarray


def main(models, song_num, output_path, files, **params):
    """

    Args:
        models:
        song_num:
        output_path:
        **params:

    Returns:
        list(dict(SDR, SIR, SAR)) <- each dump point
    """
    assert len(files) == 2
    inst1, inst2 = files

    if song_num:
        _, ref_wav = song_num_to_file(song_num, inst1)
        print(ref_wav)
        fs, signal1 = wavfile.read(ref_wav)

        _, ref_wav = song_num_to_file(song_num, inst2)
        print(ref_wav)
        fs, signal2 = wavfile.read(ref_wav)

    else:
        fs, signal1 = wavfile.read('./input/' + inst1)
        fs, signal2 = wavfile.read('./input/' + inst2)

    signal1 = preprocess(signal1, _8k)
    signal2 = preprocess(signal2, _8k)
    nch = 2

    mix = np.zeros([len(signal1), nch])

    for ch in range(nch):
        mix[:, ch] = signal1[:, ch] + signal2[:, ch]

    params.update({
        'sig1': signal1, 'sig2': signal2
    })

    sep, e = bss_idlma(mix, models, song_num, output_path, **params)

    # --- write wav ---
    output_path = '../dnn/{}/'.format(output_path)
    if not os.path.exists(output_path):
        output_path = './output'

    wavfile.write(os.path.join(output_path, 'mix_{}.wav'.format(song_num)), fsResample, mix)
    for j, f in enumerate(files):
        fname = os.path.join(output_path, '{}_{}_idlma.wav'.format(f[:-4], song_num))
        print('->', fname)
        wavfile.write(fname, fsResample, sep[j])
    print('write complete')

    # --- evaluate ---
    res = bss_eval_sources(
        reference_sources=na([signal1[:, refMic], signal2[:, refMic]]),
        estimated_sources=na([sep[0], sep[1]])
    )
    format_result(res)

    res_org = bss_eval_sources(
        reference_sources=na([signal1[:, refMic], signal2[:, refMic]]),
        estimated_sources=np.array([mix[:, refMic]] * nch)
    )
    format_result(res_org)

    sdr_imp = res[0].mean() - res_org[0].mean()
    print('SDR Improvement: {}'.format(sdr_imp))

    return e.sxr


if __name__ == '__main__':
    """
    ex)
    $ python main.py -i 100
    """
    dir_target_map = {
        'd': 'drums.wav', 'v': 'vocals.wav',
        'b': 'bass.wav', 'o': 'other.wav',
        'g': 'guitar.wav', 's': 'synth.wav',
    }

    p = argparse.ArgumentParser()
    p.add_argument('-song', '--song_num', metavar='N', type=int, default=0,
                   help='song number (default: 0 -> ./DSD100/)')
    p.add_argument('-m', '--model_path', metavar='N', type=str,
                   help='model_path( ex output -> ../dnn/outputd/model.npz')
    p.add_argument('-n', '--ns', metavar='N', type=int, default=2,
                   help='number of sources (default: 2)')
    # p.add_argument('-f', '--fftSize', metavar='N', type=int, default=1024,
    #                help=' window length in STFT (default: 1024)')
    # p.add_argument('-s', '--shiftSize', metavar='N', type=int, default=0,
    #                help='shift length in STFT (default: fftsize / 2)')
    p.add_argument('-fs', '--fsResample', metavar='N', type=int, default=8000,
                   help='fsResample (default: 16000)')
    p.add_argument('-i', '--It', metavar='N', type=int, default=100,
                   help='number of iterations (default: 50)')
    p.add_argument('-r', '--refMic', metavar='N', type=int, default=0,
                   help='refMic (default: 0)')
    p.add_argument('-d', '--dump_step', metavar='N', type=int, default=10,
                   help='dump wav and loss for each n epoch (default: 10)')
    p.add_argument('-dl', '--delta', metavar='N', type=float, default=0.01,
                   help='power + delta (default: 0.01)')
    p.add_argument('-w', '--n_update_w', metavar='N', type=int, default=10,
                   help='how many updates w par dnn predict (default: 1)')
    p.add_argument('-dr', '--draw', action='store_true', default=False,
                   help='plot cost function values or not (default: False)(todo 未実装)')
    p.add_argument('-g', '--gpu_id', metavar='N', type=int, default=0,
                   help='gpu id (-1 if use cpu)')
    p.add_argument('-semi', '--semi_supervised', action='store_true', default=False,
                   help='semi_supervised_IDLMA')
    p.add_argument('-mf', '--model_file', metavar='N', default='model', type=str,
                   help='model_filename ex)model model_10')
    p.add_argument('-s', '--dump_spectrogram', action='store_true', default=False,
                   help='dump spectrogram png')
    p.add_argument('-wf', '--wf_type', metavar='N', type=int, default=1,
                   help='winner filter type (default: 1)')
    p.add_argument('-dt', '--delta_type', metavar='N', type=str, default='max',
                   help='delta type (default: max)')
    p.add_argument('-sp', '--space_type', metavar='N', type=int, default=1,
                   help='space type')
    p.add_argument('-sf', '--suffix', metavar='N', default='', type=str,
                   help='sdr_imp_IDLMA_{output_file}{suffix}.json ')
    p.add_argument('-fls', '--files', metavar='N', type=str, default='',
                   help='use instrument ex) vocals/bass')
    args = p.parse_args()

    if not args.files:
        dirs = glob.glob('../dnn/{}/*'.format(args.model_path))
        args.files = list(map(lambda x: dir_target_map[x[-1]], filter(lambda x: len(x.split('/')[-1]) == 1, dirs)))
    else:
        args.files = list(map(lambda x: x + '.wav', args.files.split('/')))

    print('instruments:', args.files)

    models = [
        load_model('../dnn/{}/{}'.format(args.model_path, args.files[0][0]), model_name=args.model_file, gpu_id=args.gpu_id),  # output_1107dみたいな
        load_model('../dnn/{}/{}'.format(args.model_path, args.files[1][0]), model_name=args.model_file, gpu_id=args.gpu_id),
    ]

    if args.space_type == 1:
        args.files = list(map(lambda x: x[:-4] + '_E2A_pos050130_mic2123.wav', args.files))
    elif args.space_type == 2:
        args.files = list(map(lambda x: x[:-4] + '_E2A_pos050110_mic2122.wav', args.files))
    else:
        raise ValueError

    if args.song_num > 0:
        args.output_path = args.model_path
        main(models, **args.__dict__)
    else:
        args.output_path = None#args.model_path
        res = {}

        for song_num in range(301, 301 + 25):
            args.song_num = song_num
            sdr_imp = main(models, **args.__dict__)
            res[song_num] = sdr_imp

        print(res)
        IO.write_json(os.path.join('./json', 'sdr_imp_IDLMA_{}{}.json'.format(args.model_path, args.suffix)), res)

        notify.main("{}{} SDRimp:{}".format(args.model_path, args.suffix, mean_sdr_imp(res)))