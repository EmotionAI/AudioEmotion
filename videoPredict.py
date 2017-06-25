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
from keras.models import load_model
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2



# Load the model 
model_path = 'emot_speech.hdf5'
model = load_model(model_path)
class_map = {0:'Anticipation', 1:'Disgust', 2:'Fear', 3:'Anger', 4:'Surprise', 5:'Joy', 6:'Sadness'}
#Intitialize parameters
T_per_example = 5
window_length = 0.03
window_step = window_length/2.0
num_cep_coeffs = 13
out_dir = 'Output/'
#print sys.argv
sound_file = sys.argv[1] + '.wav'#'test/v1H.wav'
vid_file = sys.argv[1] + '.mp4'

out_file = out_dir + sys.argv[1].split('/')[-1] + '_out'+'.mp4'



fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out1 = cv2.VideoWriter(out_file,fourcc, 30.0, (1920,1080), False)
#rgbVideo = cv2.VideoCapture('./videos/MAH00205.MP4')
#rgbVideo = cv2.VideoCapture('./videos/MAH00206.MP4')
rgbVideo = cv2.VideoCapture(vid_file)


data_list = []

# Function to calculate energy
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
# Main Code
rate,wavdata = wavfile.read(sound_file)
wavdata = wavdata.astype('float32')
#bp()
if len(wavdata.shape) > 1:
	wavdata = wavdata[:,0]
samp_per_example = int(math.floor(rate*T_per_example))
num_intervals = int(len(wavdata)/samp_per_example)
#bp()
for i in range(0, num_intervals*samp_per_example, samp_per_example):
	feature1 = mfcc(wavdata[i: i+samp_per_example], samplerate=rate, winlen=window_length, winstep=window_step, numcep = num_cep_coeffs)
	feature2 = calc_features(wavdata[i: i+samp_per_example], samplerate = rate, samp_per_example = samp_per_example, window_step = window_step)
	feature2 = feature2.reshape((-1,1))
	feature = np.concatenate([feature1, feature2], axis = 1)
	data_list.append(feature)
	
data_arr = np.array(data_list)
data_arr = data_arr[:,:,:-1]
#np.save('testing_video', data_list)
data_flat = data_arr.reshape((-1,13))
featuremean = np.mean(data_flat, axis = 0)
featurestd = np.std(data_flat, axis = 0)
for i in range(len(data_arr)):
	data_arr[i,:,:] -= featuremean
	data_arr[i,:,:] /= featurestd
	
output = model.predict(data_arr)


correct = 5
numCorr = 0
for t in output:
	print map(lambda x:round(x,2),t),	class_map[np.argmax(t)]
	#t=map(lambda x:round(x,2),t)
	#print t
	if np.argmax(t)==correct:
		numCorr+=1#t[correct]
print numCorr, output.shape,output.shape[0]
print 'acc ', numCorr*1.0/output.shape[0]



done=0
for t in output:
    for i in range(150):
    	ret, frame = rgbVideo.read()
    	if not ret:
    		done=1
    		break
    	emotion=class_map[np.argmax(t)]
    	cv2.putText(frame, emotion, (7,7), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    	out1.write(emotion, frame)
    if done:
    	break

out1.release()

# filename = '/media/vaibhav/7A30-DFE2/Anticipation/WilkAnticipation.wav'
# rate,wavdata = wavfile.read(filename)
# print wavdata.shape
# print rate
