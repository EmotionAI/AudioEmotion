import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from keras import regularizers
#import sklearn
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from pdb import set_trace as bp

np.random.seed(100)
model_path = 'emot_speech.hdf5'

featmean = np.load('norm_mean.npy')[:-1]
featstd = np.load('norm_std.npy')[:-1]



data = {}
data['Anticipation'] = np.load('Anticipation.npy')[:,:,:-1]
data['Joy'] = np.load('Joy.npy')[:,:,:-1]
data['Disgust'] = np.load('Disgust.npy')[:,:,:-1]
data['Anger'] = np.load('Anger.npy')[:,:,:-1]
data['Surprise'] = np.load('Surprise.npy')[:,:,:-1]
data['Sadness'] = np.load('Sadness.npy')[:,:,:-1]
data['Fear'] = np.load('Fear.npy')[:,:,:-1]


class_size = {}


featureSum = np.zeros(7)

for key in data:
	np.random.shuffle(data[key])
	data[key] -= featmean
	data[key] /= featstd
	class_size[key] = data[key].shape[0]


YDim = 7
HidDim1 = 64
HidDim2 = 64
T = 333
XDim = 13

data_dir = '/media/vaibhav/7A30-DFE2/'
class_map = {'Anticipation':0, 'Disgust':1, 'Fear': 2, 'Anger': 3, 'Surprise': 4, 'Joy': 5, 'Sadness': 6}



# Define Model
model = Sequential()
model.add(LSTM(HidDim1, input_shape = (None,XDim), activation = 'tanh', return_sequences = True))
model.add(LSTM(HidDim2, input_shape = (T,HidDim1), activation = 'tanh'))

model.add(Dense(YDim, input_dim = HidDim2, activation='softmax', name="hid"))#, kernel_regularizer=regularizers.l2(0.0)))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


numEpochs = 10 * 150
numIter = 100
ex_per_class = 18
batchsize = 128

best_loss = np.inf

for epoch in range(numEpochs):
	# Create batch
	avg_loss, avg_acc = 0, 0
	for i in range(numIter):
		X_batch = []
		Y_batch = []
		for cls in data:
			idx = np.random.randint(data[cls].shape[0] - 100, size = ex_per_class)
			X_batch.append(data[cls][idx,:,:])
			label = np.zeros((ex_per_class, YDim))
			label[:, class_map[cls]] = 1
			Y_batch.append(label)
		Y_batch = np.concatenate(Y_batch)
		X_batch = np.concatenate(X_batch)		
		loss, acc = model.train_on_batch(X_batch, Y_batch)
		avg_loss += loss*1.0/numIter
		avg_acc += acc*1.0/numIter
		


	print 'epoch {0:.4f}, loss {1:.4f}, acc {2:.4f}'.format(epoch, avg_loss, avg_acc)
	if best_loss > avg_loss:

		print "loss decreased from {0:.4f} to {1:.4f}, saving weights".format(best_loss, avg_loss)
		best_loss = avg_loss
		model.save(model_path)











