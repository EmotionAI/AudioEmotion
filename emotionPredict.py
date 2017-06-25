
# coding: utf-8

# In[7]:




# In[ ]:

import numpy as np
from sklearn.metrics import roc_curve, auc
from pdb import set_trace as bp
from keras.models import load_model


# Load the model 
model_path = 'emot_speech.hdf5'
model = load_model(model_path)

# Load the normalizers
np.random.seed(100)
featmean = np.load('norm_mean.npy')[:-1]
featstd = np.load('norm_std.npy')[:-1]

# Load the data
data = {}
data['Anticipation'] = np.load('Anticipation.npy')[:,:,:-1]
data['Joy'] = np.load('Joy.npy')[:,:,:-1]
data['Disgust'] = np.load('Disgust.npy')[:,:,:-1]
data['Anger'] = np.load('Anger.npy')[:,:,:-1]
data['Surprise'] = np.load('Surprise.npy')[:,:,:-1]
data['Sadness'] = np.load('Sadness.npy')[:,:,:-1]
data['Fear'] = np.load('Fear.npy')[:,:,:-1]

# Store the class size
class_size = {}
num_classes = 7
featureSum = np.zeros(num_classes)

# Shuffle the data in the same way that the train data was shuffled

for key in data:
    np.random.shuffle(data[key])
    data[key] -= featmean
    data[key] /= featstd
    class_size[key] = data[key].shape[0]
# Use the last 100 points as the test data
test_data = {}
# This stores the confusion matrix
error_data = {}
for cls in data:
    temp = data[cls]
    test_data[cls] = temp[-100:] 
    t = model.predict(test_data[cls])
  
    error_data[cls] = map( lambda x: round(x,2) , sum(t)/100.0)
    print error_data[cls]#"For class {}, error is {}".format(cls,error_data[cls])





# In[29]:



