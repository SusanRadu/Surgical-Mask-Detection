#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa # library that will help read the .wav files
from glob import glob # library that will help read the names of the files in folders
import matplotlib.pyplot as plt # library that helps ploting
import numpy as np # library for mathematical operations and more
import scipy as sp # library for converting the aplitude array with fft


# In[2]:


f = open(r'C:\Users\RADU\Desktop\data\train.txt', "r")# open the file where the audio train data name and label is
lines = f.readlines()# read the lines

data = []
labels = []
train_labels = []#array that will contain the training data labels, in order
for i in range(8000):
    s = lines[i][0] + lines[i][1] + lines[i][2] + lines[i][3] + lines[i][4]+ lines[i][5]# the first 6 chars in a line are the name of the audio file
    x = int(s)
    y = int(lines[i][11])#the twelvth char in a line is the label number
    data.append(x)
    labels.append(y)
    
train = np.stack((data, labels), axis=1)#combine the two list into an matrix with (name,label)
train = train[np.argsort(train[:, 0])]#sort the matrix by audio file names

for i in range(8000):
    train_labels.append(train[i][1])#extract the labels of the train data 
    
f.close()
    


# In[20]:


f = open(r'C:\Users\RADU\Desktop\data\validation.txt', "r")
lines = f.readlines()
#same thing as above except for validation data labels
data = []
labels = []
validation_labels = []
for i in range(1000):
    s = lines[i][0] + lines[i][1] + lines[i][2] + lines[i][3] + lines[i][4]+ lines[i][5]
    x = int(s)
    y = int(lines[i][11])
    data.append(x)
    labels.append(y)
    
train = np.stack((data, labels), axis=1)
train = train[np.argsort(train[:, 0])]

for i in range(1000):
    validation_labels.append(train[i][1])
    
f.close()


# In[4]:


train_data = []#list that will contain arrays of aplitudes of the audio files
audio_files = glob(r'C:\Users\RADU\Desktop\data\train' + '/*.wav')#get the path of every audio file in the train data folder
for i in range(8000):
    data, sampling_rate = librosa.load(audio_files[i],sr=None,mono=True,offset=0.0,duration=None)#load the audio files
    train_data.append(data)


# In[5]:


validation_data = []#same thing as above for validation data
audio_files = glob(r'C:\Users\RADU\Desktop\data\validation' + '/*.wav')
for i in range(1000):
    data, sampling_rate = librosa.load(audio_files[i],sr=None,mono=True,offset=0.0,duration=None)
    validation_data.append(data)


# In[6]:


test_data = []#same thing as above for testing data
audio_files = glob(r'C:\Users\RADU\Desktop\data\test' + '/*.wav')
for i in range(3000):
    data, sampling_rate = librosa.load(audio_files[i],sr=None,mono=True,offset=0.0,duration=None)
    test_data.append(data)


# In[30]:


test_fft[0].shape


# In[7]:


train_fft = []#train_data converted with fast fourier transform
validation_fft = []
test_fft = []

for i in range(8000):
    train_fft.append(abs(sp.fft.fft(train_data[i])))#scipy fft conversion

for i in range(1000):
    validation_fft.append(abs(sp.fft.fft(validation_data[i])))

for i in range(3000):
    test_fft.append(abs(sp.fft.fft(test_data[i])))


# In[8]:


train_mfcc = []#train_data converted to mel-frequency cepstral coefficients
test_mfcc = []
validation_mfcc = []

for i in range(8000):
    a = librosa.feature.mfcc(train_data[i],sampling_rate)#librosa mfcc convert function
    a = a.reshape(-1)
    train_mfcc.append(a)

for i in range(1000):
    a = librosa.feature.mfcc(validation_data[i],sampling_rate)
    a = a.reshape(-1)
    validation_mfcc.append(a)

for i in range(3000):
    a = librosa.feature.mfcc(test_data[i],sampling_rate)
    a = a.reshape(-1)
    test_mfcc.append(a)


# In[9]:


#train_fft = []
#for i in range(1000):
#    train_fft.append(np.log(abs(librosa.core.stft(validation_data[i]))))


# In[10]:


train_spect = []
for i in range(8000):
    a,b,c = sp.signal.spectrogram(train_data[i],sampling_rate)#scipy spectrogram conversion function
    train_spect.append(c.reshape(-1))#we have to make the returned matrix into an array, so we can use it in our model


# In[11]:


validation_spect = []
for i in range(1000):
    a,b,c = sp.signal.spectrogram(validation_data[i],sampling_rate)
    validation_spect.append(c.reshape(-1))


# In[56]:


from sklearn.naive_bayes import MultinomialNB#the naive bayes model from sklearn

naive_bayes_model = MultinomialNB()#create the model
naive_bayes_model.fit(train_fft, train_labels)#train the model
y_pred_nb = naive_bayes_model.predict(validation_fft)#the results from validation prediction


# In[68]:


import sklearn.svm as sk#the support vector machine model from sklearn

svc = sk.SVC(C=4)#create the model
svc.fit(train_fft,train_labels)#train the model
y_pred_svm = svc.predict(validation_fft)#the results from validation prediction


# In[82]:


#block of code that will score our validation prediction
k=0
for i in range(1000):
    if y_pred_svm[i]==validation_labels[i]:
        k=k+1


# In[83]:


confusion_matrix = np.zeros((2,2))
for i, y in enumerate(y_pred_svm):
        confusion_matrix[validation_labels[i]][y] += 1
confusion_matrix


# In[57]:


k=0
for i in range(1000):
    if y_pred_nb[i]==validation_labels[i]:
        k=k+1
print('Accuracy',k,k/1000)
print('Loss',1000-k,1-(k/1000))


# In[58]:


confusion_matrix = np.zeros((2,2))
for i, y in enumerate(y_pred_nb):
        confusion_matrix[validation_labels[i]][y] += 1
confusion_matrix


# In[ ]:


#block of code that will score our validation prediction
k=0
for i in range(1000):
    if y_pred_nb[i]==validation_labels[i]:
        k=k+1
print('Accuracy',k,k/1000)
print('Loss',1000-k,1-(k/1000))


# In[15]:


#block of code that will write the test prediciton into an csv named submission
f = open(r'C:\Users\RADU\Desktop\data\submission.txt','w+')

f.write("name,label")
f.write("\n")
for i in range(3000):
    x = 300001+i
    s = str(x)
    f.write(s + ".wav")
    f.write(","+str(y_pred[i]))
    f.write("\n")




