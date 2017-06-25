import numpy as np
from pdb import set_trace as bp

data = {}
data['Anticipation'] = np.load('Anticipation.npy')
data['Joy'] = np.load('Joy.npy')
data['Disgust'] = np.load('Disgust.npy')
data['Anger'] = np.load('Anger.npy')
data['Surprise'] = np.load('Surprise.npy')
data['Sadness'] = np.load('Sadness.npy')
data['Fear'] = np.load('Fear.npy')


class_size = {}


featureSum = np.zeros(7)

data = np.concatenate([data['Anticipation'], data['Joy'], data['Disgust'], data['Anger'], data['Surprise'], data['Sadness'], data['Fear']])
data = data.reshape((-1,14))
featuremean = np.mean(data, axis = 0)
featurestd = np.std(data, axis = 0)


np.save('norm_std', featurestd)
np.save('norm_mean', featuremean)