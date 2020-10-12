import os
import numpy as np
import argparse
from scipy.io import wavfile
#from ilrma import bss_ilrma
from fdica_make_SPEC import bss_fdica
from mir_eval.separation import bss_eval_sources
import glob
import random
import pickle
import multiprocessing
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', True)

# Setting parameters
# seed = 4  # seed of pseudo random values
refMic = 0  # reference mic for evaluation
fsResample = 16000  # resampling frequency
na = np.asarray
def main(point):
    start = point*10
    end = start+10
    for i in np.arange(start,end):
        try:
            #reverberation_file_list = glob.glob('./make_dnn_models/input/reverberation_wavfile470ms/train/'+str(random.randint(1, reverberation_file_num))+'/*.wav')  # カット前
            reverberation_file_num = len(glob.glob('./make_dnn_models/10s_wav_JR2_conv/*'))  # カット前
            a = glob.glob('./make_dnn_models/10s_wav_JR2_conv/'+str(4)+'/*pos060_mic2123_conv.wav')  # カット前
            #reverberation_file_list = glob.glob('./make_dnn_models/input/reverberation_wavfile470ms/train/'+str(random.randint(1, reverberation_file_num/2))+'/*.wav')  # カット前

            #b = glob.glob('./make_dnn_models/input/reverberation_wavfile470ms/train/'+str(random.randint(1, reverberation_file_num/2)) +'/*pos060_mic21.wav')[0]
            #target = random.randint(1,2)#class数
            target =0
            if target ==0:#男女別
                #filenumber = random.randint(1, reverberation_file_num)
                #fs, signals1m1 = wavfile.read(a[0])
                fs, signals1 = wavfile.read(glob.glob('./make_dnn_models/10s_wav_JR2_conv/'+str(i)+'/*pos060_mic2123_conv.wav')[0])
                signals1 = signals1 / 32768.0
                assert fs == fsResample
                fs, signals2 = wavfile.read(glob.glob('./make_dnn_models/10s_wav_JR2_conv/'+str(i)+'/*pos120_mic2123_conv.wav')[0])
                signals2 = signals2 / 32768.0
                assert fs == fsResample


                signals1m1 =signals1[:,0]
                signals1m2 =signals1[:,1]
                signals2m1 =signals2[:,0]
                signals2m2 =signals2[:,1]


            nch = 2
            mix = np.zeros([len(signals1m1), nch])
            origin = np.zeros([len(signals1m1), nch])

            mix[:, 0] = signals1m1 + signals2m1
            mix[:, 1] = signals1m2 + signals2m2

            origin[:, 0] = signals1m1
            origin[:, 1] = signals2m1

            """変更"""
            #sep, cost = bss_fdica(mix,origin, **params)
            #bss_fdica(mix,origin,i, **params)
            bss_fdica(mix,origin,i)
        except:
            print('no dir')


if __name__ == '__main__':
    """
    ex)
    $ python main.py -b 10
    """

    #p = Pool(4)
    #p.map( main, range(10) )#nijouに0,1,..のそれぞれを与えて並列演算
    #a = np.arange(0,10)
    print(multiprocessing.cpu_count())
    #main(1)

    p = Pool(10)
    p.map( main, np.arange(0,19) )#nijouに0,1,..のそれぞれを与えて並列演算
