#!/usr/bin/env python
# coding: utf-8

# In[7]:


cd D:\User-Majid\Projects\DeepProtein


# In[27]:


import pickle
import random
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np


# In[82]:


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer import Adam
form


# In[29]:


f = open("data.pkl", 'rb')
rawProt = pickle.load(f)
f.close()


# In[31]:


with open("labels.pkl", 'rb') as f:
  labels = pickle.load(f)

labels = np.reshape(labels, 400000)


# In[32]:


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


# In[33]:


#add space for the tokenizer's seperator
def add_space(string):
  return ' '.join(string)

for i in range(len(ppd)):
  ppd[i] = add_space(ppd[i])


# In[34]:


alphabet = 'A C D E F G H I K L M N P Q R S T V W Y'
t = Tokenizer(num_words=20, split=" ")
t.fit_on_texts([alphabet])
seq = t.texts_to_sequences(ppd)
type(seq)


# In[35]:


from keras.preprocessing.sequence import pad_sequences
ppd = pad_sequences(seq, padding='post')


# In[40]:


del seq


# In[66]:


#split train and test data set
#will have 266457 train data and 66614 test data
#could not use sklearn.model_selection.train_test_split because of memory
X_train, X_test = ppd[:266458], ppd[266458:]
y_train, y_test = labels[:266458], labels[266458:]


# In[54]:


def one_hotter(arr):
    hot = np.zeros((12681, 20))
    for i in range(12681):
        if arr[i] != 0:
            hot[i, arr[i]-1] = 1
    return hot.flatten()


def generator(array, labels, batch_size=10):
     i = 0
    while i <= 266457
        subarray = array[i: i+batch_size]
        sublabels = labels[i: i+batch_size]
        one_hotted = np.apply_along_axis(one_hotter, axis=1, arr=subarray)
        i = i + batch_size
        yield one_hotted, sublabels 


# In[81]:


ht = np.zeros((12681, 20))
tst = X_train[0:300, :]  # 300 is the maximum number of elements that could be preprocessed without running out of memory
a = np.apply_along_axis(one_hotter, axis=1, arr=tst)
print(a[4, 7, :], a.shape)
print(X_train[4, 7])


# In[51]:


#creating the model:
def create_model():
    model = Sequential()
    model.add(Dense(500, input_dim=253620, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


#fit the model
BatchGenerator = generator(X_train, y_train, batch_size=50)
model = create_model()
c = 0
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
fig.tight_layout()
for X_trainbatch, y_trainbatch in BatchGenerator:
    c += 1
    print("batch %d" % c)
    history = model.fit(X_trainbatch, y_trainbatch, batch_size=10, epochs=2, validation_split=0.1, verbose=1, shuffle=1)
    
    axs[0, 0].plot(range(2*(c-1), 2*c), history.history['loss'], 'r')
    axs[0, 0].plot(range(2*(c-1), 2*c), history.history['val_loss'], 'b')
    axs[0, 0].set_title("Loss")
    axs[0, 0].legend(['loss', 'val_loss'])
    axs[0, 0].set_xlabel('epoch')
    axs[0, 0].set_ylabel('loss')
    
    axs[1, 0].plot(range(2*(c-1), 2*c), history.history['acc'], 'r')
    axs[1, 0].plt.plot(range(2*(c-1), 2*c), history.history['val_acc'], 'b')
    axs[1, 0].set_title("Accuracy")
    axs[1, 0].legend(['acc', 'val_acc'])
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].set_ylabel('Accuracy')

