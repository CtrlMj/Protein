#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', 'D:\\User-Majid\\Projects\\DeepProtein')


# In[2]:


import pickle
import random
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np


# In[20]:


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# In[4]:


f = open("data.pkl", 'rb')
rawProt = pickle.load(f)
f.close()


# In[5]:


with open("labels.pkl", 'rb') as f:
  labels = pickle.load(f)

labels = np.reshape(labels, 400000)


# In[6]:


prots = np.reshape(rawProt, 400000)
#Here Im gonna remove the duplicates in the following pythonic way: unique = np.unique(array, axis=0)
print(len(np.unique(prots)))
prots, indices = np.unique(prots, return_index=True)
labels = labels[indices]
print(len(labels))
#now remove the ones containing
removals = ['B', 'J', 'O', 'U', 'X', 'Z']
lst = []
y = []
for i in range(len(prots)):
  if not(any(toberemoved in prots[i] for toberemoved in removals)):
    lst.append(prots[i])
    y.append(labels[i])
 
print(len(lst))
ppd = np.array(lst, dtype=object) #ppd for Polished Protein Data
labels = np.array(y, dtype=np.uint8)
del y
print(ppd.shape)
print(labels.dtype)


# In[7]:


ones = np.where(labels == 1)[0]
zeros = np.where(labels == 0)[0]
s = np.random.choice(287467, 45000)
labelsones = labels[ones]
labelszeros = labels[zeros]
labelszeros = labelszeros[s]
ppdones = ppd[ones]
ppdzeros = ppd[zeros]
ppdzeros = ppdzeros[s]


# In[8]:


labels = np.concatenate((labelsones, labelszeros), axis=0)
ppd = np.concatenate((ppdones, ppdzeros), axis=0)
from sklearn.utils import shuffle
ppd, labels = shuffle(ppd, labels, random_state=0)


# In[9]:


#add space for the tokenizer's seperator
def add_space(string):
  return ' '.join(string)

for i in range(len(ppd)):
  ppd[i] = add_space(ppd[i])


# In[10]:


alphabet = 'A C D E F G H I K L M N P Q R S T V W Y'
t = Tokenizer(num_words=20, split=" ")
t.fit_on_texts([alphabet])
seq = t.texts_to_sequences(ppd)
type(seq)


# In[11]:


print(t.word_index)


# In[12]:


from keras.preprocessing.sequence import pad_sequences
ppd = pad_sequences(seq, padding='post')


# In[13]:


print(type(ppd), ppd.shape)
del seq


# In[14]:


#split train and test data set
#will have 90000 train data and 66614 test data
#could not use sklearn.model_selection.train_test_split because of memory
X_train, X_test = ppd[:36000], ppd[36000:]
y_train, y_test = labels[:36000], labels[36000:]


# In[15]:


def one_hotter(arr):
    hot = np.zeros((10553, 20))
    for i in range(10553):
        if arr[i] != 0:
            hot[i, arr[i]-1] = 1
    return hot.flatten()


def generator(array, labels, batch_size=10):
    i = 0
    while i < 36000:
        subarray = array[i: i+batch_size]
        sublabels = labels[i: i+batch_size]
        one_hotted = np.apply_along_axis(one_hotter, axis=1, arr=subarray)
        i = i + batch_size
        yield one_hotted, sublabels 


# In[16]:


tst = X_train[0:300, :]  # around 250-200 is the maximum number of elements that could be preprocessed without running out of memory
a = np.apply_along_axis(one_hotter, axis=1, arr=tst)
print(a[4, :23], a.shape)
print(X_train[4, :2])


# In[17]:


#One hot encode the test data - we pick a small subset of that data to avoid memory error
X_test = np.apply_along_axis(one_hotter, axis=1, arr=X_test[-300:])
y_test = y_test[-300:]


# In[23]:


#creating the model:
def create_model():
    model = Sequential()
    model.add(Dense(33, input_dim=211060, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(33, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(33, activation='elu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[24]:


#fit the model
model = create_model()
print(model.summary())
c = 0
Accuracy = []
val_Accuracy = []
Loss = []
val_Loss = []
epochs = 2
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
fig.tight_layout()
pltcntr = 0
for epoch in range(epochs):
    BatchGenerator = generator(X_train, y_train, batch_size=100)
    print("EPOCH", epoch)
    c = 0
    for X_trainbatch, y_trainbatch in BatchGenerator:
        c += 1
        pltcntr += 1
        print(" BATCH %d" % c)
        history = model.train_on_batch(X_trainbatch, y_trainbatch)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Loss =", history[0], "Accuracy =", history[1], "Val_Accuracy =", scores[1])
        Loss.append(history[0])
        val_Loss.append(scores[0])
        Accuracy.append(history[1])
        val_Accuracy.append(scores[1])
    print("=======================================================================")

#axs[0].scatter(pltcntr, history[0], color='red', s=1)
axs[0].plot(Loss, 'r')
axs[0].plot(val_Loss, 'b')
axs[0].set_title("Loss")
axs[0].legend(['loss', 'val_loss'])
axs[0].set_xlabel('batch-epoch')
axs[0].set_ylabel('loss')

#axs[1].scatter(pltcntr, history[1], color='blue', s=1)
axs[1].plot(Accuracy, 'r')
axs[1].plot(val_Accuracy, 'b')
axs[1].set_title("Accuracy")
axs[1].legend(['acc', 'val_accuracy'])
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('Accuracy')
plt.show()


# In[ ]:





# In[ ]:




