import numpy as np
from scipy import stats
from scipy.io import wavfile 
import pysptk as sptk

#classes = ['Anger','Surprise','Sad','Joy','Disgust']
#for cls in classes:
filename = 'test.wav'
rate,wavdata = wavfile.read(filename)

print(len(wavdata)) 
