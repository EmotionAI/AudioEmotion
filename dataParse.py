from scipy import stats
from scipy.io import wavfile
import pysptk as sptk
from os import listdir
import math
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pdb import set_trace as bp
import numpy as np

T_per_example = 5
window_length = 0.03
window_step = window_length/2.0
num_cep_coeffs = 13

data_dir = '/media/vaibhav/7A30-DFE2/'
class_map = {'Anticipation':0, 'Disgust':1, 'Fear': 2, 'Anger': 3, 'Surprise': 4, 'Joy': 5, 'Sadness': 6}


def calc_features(signal, samplerate, samp_per_example, window_step):
	#pitch = []
	energy = []
	interval = int(math.floor(samplerate*window_step))
	signal = signal.copy(order = 'C')
	num_intervals = int(len(signal)/interval)
	for i in range(0, num_intervals*interval, interval):
		#pitch.append(sptk.rapt(signal[i:i+interval*2], samplerate, hopsize = interval/2, otype = 0))
		#temp = sptk.rapt(signal[i:i+interval*2], samplerate, hopsize = interval/100, otype = 0)
		#print temp.shape
		energy.append(np.mean(np.square(signal[i:i+interval*2])))
	#return np.concat([np.array(pitch), np.array(energy)])
	return np.array(energy)


"""

0:12 --- MFCC
13   --- pitch
14   --- energy

"""

for cls in ['Sadness']:
	files = listdir(data_dir+cls)
	data_list = []
	for file in files:
		print file
		rate,wavdata = wavfile.read(data_dir+cls + '/' + file)
		wavdata = wavdata.astype('float32')
		#bp()
		if len(wavdata.shape) > 1:
			wavdata = wavdata[:,0]
		samp_per_example = int(math.floor(rate*T_per_example))
		num_intervals = int(len(wavdata)/samp_per_example)
		for i in range(0, num_intervals*samp_per_example, samp_per_example):
			feature1 = mfcc(wavdata[i: i+samp_per_example], samplerate=rate, winlen=window_length, winstep=window_step, numcep = num_cep_coeffs)
			feature2 = calc_features(wavdata[i: i+samp_per_example], samplerate = rate, samp_per_example = samp_per_example, window_step = window_step)
			feature2 = feature2.reshape((-1,1))
			feature = np.concatenate([feature1, feature2], axis = 1)
			data_list.append(feature)
	data_list = np.array(data_list)
	np.save(cls, data_list)




# filename = '/media/vaibhav/7A30-DFE2/Anticipation/WilkAnticipation.wav'
# rate,wavdata = wavfile.read(filename)
# print wavdata.shape
# print rate
