#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Embedding network:
get_ipython().run_line_magic('cd', 'D:\\User-Majid\\Projects\\DeepProtein')


# In[2]:


import pickle
import random
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.layers import Flatten


# In[3]:


f = open("data.pkl", 'rb')
rawProt = pickle.load(f)
f.close()

with open("labels.pkl", 'rb') as f:
  labels = pickle.load(f)

labels = np.reshape(labels, 400000)


# In[4]:


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


# In[5]:


def add_space(string):
  return ' '.join(string)

for i in range(len(ppd)):
  ppd[i] = add_space(ppd[i])

alphabet = 'A C D E F G H I K L M N P Q R S T V W Y'
t = Tokenizer(num_words=20, split=" ")
t.fit_on_texts([alphabet])
seq = t.texts_to_sequences(ppd)
type(seq)
print(seq[:2])


# In[6]:


from keras.preprocessing.sequence import pad_sequences
ppd = pad_sequences(seq, padding='post')
print(type(ppd))
del seq
#split train and test data set
#will have 266457 train data and 66614 test data
#could not use sklearn.model_selection.train_test_split because of memory
X_train, X_test = ppd[:266458], ppd[266458:]
y_train, y_test = labels[:266458], labels[266458:]


# In[7]:


def generator(array, labels, batch_size=10):
    i = 0
    while i <= 266457:
        subarray = array[i: i+batch_size]
        sublabels = labels[i: i+batch_size]
        i = i + batch_size
        yield subarray, sublabels 


# In[20]:


def create_model():
    model = Sequential()
    model.add(Embedding(21, 5, input_length=12681))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[21]:


BatchGenerator = generator(X_train, y_train, batch_size=100)
model = create_model()
print(model.summary())
c = 0
epochs = 3
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))
fig.tight_layout()
pltcntr = 0
for epoch in range(epochs):
    BatchGenerator = generator(X_train, y_train, batch_size=100)
    print("EPOCH", epoch)
    c = 0
    for X_trainbatch, y_trainbatch in BatchGenerator:
        c, pltcntr = c + 1, pltcntr + 1
        print(" BATCH %d" % c, "plt:", pltcntr)
        history = model.train_on_batch(X_trainbatch, y_trainbatch)
        print("Loss =", history[0], "Accuracy =", history[1])
        axs[0].scatter(pltcntr, history[0], color='red', s=1)
        #axs[0].plot(range(epoch*(c-1), 2*epoch), history.history['val_loss'], 'b')
        axs[0].set_title("Loss")
        axs[0].legend(['loss'])
        axs[0].set_xlabel('batch-epoch')
        axs[0].set_ylabel('loss')

        axs[1].scatter(pltcntr, history[1], color='blue', s=1)
        #axs[1].plot(range(epoch*(pltcntr-1), pltcntr*epoch), history.history['val_acc'], 'b')
        axs[1].set_title("Accuracy")
        axs[1].legend(['acc'])
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('Accuracy')
    print("=======================================================================")

plt.show()


# In[ ]:




