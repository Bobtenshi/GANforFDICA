import os
import numpy as np
import argparse
from scipy.io import wavfile
from pyILRMA.ilrma import bss_ilrma
from mir_eval.separation import bss_eval_sources
import numpy as np
import argparse
from dnn.make_datasets import preprocess, files
from dnn.evaluate import song_num_to_file
import io_handler as IO
from stft import stft
from dust.res import mean_sdr_imp
import notify
from separation.evaluater import Evaluate

# Setting parameters
# seed = 4  # seed of pseudo random values
refMic = 0  # reference mic for evaluation
fsResample = 8000  # resampling frequency

# space_type = 1
# if space_type == 1:
#     files = list(map(lambda x: x[:-4] + '_E2A_pos050130_mic2123.wav', files))
# elif space_type == 2:
#     files = list(map(lambda x: x[:-4] + '_E2A_pos050110_mic2122.wav', files))
# else:
#     raise ValueError


def format_result(res, print_perm=True):
    print('---------------')
    print('SDR: {}'.format(' '.join(map(str, res[0]))))
    print('SIR: {}'.format(' '.join(map(str, res[1]))))
    print('SAR: {}'.format(' '.join(map(str, res[2]))))
    if print_perm:
        print('perm: {}'.format(' '.join(map(str, res[3]))))

na = np.asarray


def main(song_num, files, output_path='./ILRMA', **params):
    assert len(files) == 2
    inst1 = files[0]
    inst2 = files[1]

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

    signal1 = preprocess(signal1)
    signal2 = preprocess(signal2)
    nch = 2

    mix = np.zeros([len(signal1), nch])

    for ch in range(nch):
        mix[:, ch] = signal1[:, ch] + signal2[:, ch]

    params.update({
        'sig1': signal1, 'sig2': signal2
    })

    refMic = 0
    S1, _ = stft(signal1[:, refMic], params['fftSize'], params['shiftSize'])  # for debug
    S2, _ = stft(signal2[:, refMic], params['fftSize'], params['shiftSize'])  # for debug
    e = Evaluate([signal1, signal2], [S1, S2], song_num, output_path, shiftSize=params['shiftSize'])

    sep, e = bss_ilrma(mix, e=e, **params)

    # --- write wav ---
    if output_path is not None:
        output_path = '../dnn/{}/'.format(output_path)
        if not os.path.exists(output_path):
            output_path = './output'

        wavfile.write(os.path.join(output_path, 'mix_{}.wav'.format(song_num)), fsResample, mix)
        for j, f in enumerate(files):
            fname = os.path.join(output_path, '{}_{}_ilrma.wav'.format(f[:-4], song_num))
            print('->', fname)
            wavfile.write(fname, fsResample, sep[j][:, refMic])
        print('write complete')

    # --- evaluate ---
    res = bss_eval_sources(
        reference_sources=na([signal1[:, refMic], signal2[:, refMic]]),
        estimated_sources=na([sep[0][:, refMic], sep[1][:, refMic]])
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
    $ python main.py -b 10
    """
    dir_target_map = {
        'd': 'drums.wav', 'v': 'vocals.wav',
        'b': 'bass.wav', 'o': 'other.wav',
        'g': 'guitar.wav', 's': 'synth.wav',
    }

    p = argparse.ArgumentParser()
    p.add_argument('-song', '--song_num', metavar='N', type=int, default=0,
                   help='song number (default: 0 -> all)')
    p.add_argument('-n', '--ns', metavar='N', type=int, default=2,
                   help='number of sources (default: 2)')
    p.add_argument('-f', '--fftSize', metavar='N', type=int, default=4096,
                   help=' window length in STFT (default: 4096)')
    p.add_argument('-s', '--shiftSize', metavar='N', type=int, default=2048,
                   help='shift length in STFT (default: 2048)')
    p.add_argument('-i', '--it', metavar='N', type=int, default=200,
                   help='number of iterations (default: 200)')
    p.add_argument('-b', '--nb', metavar='N', type=int, default=20,
                   help='number of bases (default: 20)')
    p.add_argument('-t', '--type', metavar='N', type=int, default=1,
                   help='1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)(todo type2未実装)')
    p.add_argument('-d', '--dump_step', metavar='N', type=int, default=10,
                   help='dump wav and loss for each n epoch (default: 10)')
    p.add_argument('-sp', '--space_type', metavar='N', type=int, default=1,
                   help='space type (see above)')
    p.add_argument('-fls', '--files', metavar='N', type=str, default='vocals/bass',
                   help='use instrument ex) vocals/bass')
    p.add_argument('-sf', '--suffix', metavar='N', default='', type=str,
                   help='sdr_imp_ILRMA_{suffix}.json ')
    args = p.parse_args()

    args.files = list(map(lambda x: x + '.wav', args.files.split('/')))

    if args.space_type == 1:
        args.files = list(map(lambda x: x[:-4] + '_E2A_pos050130_mic2123.wav', args.files))
    elif args.space_type == 2:
        args.files = list(map(lambda x: x[:-4] + '_E2A_pos050110_mic2122.wav', args.files))
    else:
        raise ValueError

    print('instruments:', args.files)

    if args.song_num > 0:
        main(**args.__dict__)
    else:
        print('all')
        args.output_path = './output/ilrma'
        res = {}

        song_start_index = 301

        for i in range(1):
            for song_num in range(song_start_index, song_start_index + 25):
                args.song_num = song_num
                sdr_imp = main(**args.__dict__)
                res[song_num] = sdr_imp

            print(res)
            filepath = os.path.join('json', 'sdr_imp_ILRMA_{}.json'.format(args.suffix))

            IO.remove_if_exists(filepath)
            IO.write_json(filepath, res)

            notify.main("ILRMA {} SDRimp:{}".format(args.suffix, mean_sdr_imp(res)))