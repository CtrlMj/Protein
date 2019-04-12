#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Embedding network:
get_ipython().run_line_magic('cd', 'D:\\User-Majid\\Projects\\DeepProtein')


# In[9]:


import pickle
import random
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.layers import Flatten


# In[10]:


f = open("data.pkl", 'rb')
rawProt = pickle.load(f)
f.close()

with open("labels.pkl", 'rb') as f:
  labels = pickle.load(f)

labels = np.reshape(labels, 400000)


# In[11]:


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
del lst
print(ppd.shape)
print(labels.dtype)


# In[12]:


def add_space(string):
  return ' '.join(string)

for i in range(len(ppd)):
  ppd[i] = add_space(ppd[i])

alphabet = 'A C D E F G H I K L M N P Q R S T V W Y'
t = Tokenizer(num_words=20, split=" ")
t.fit_on_texts([alphabet])
seq = t.texts_to_sequences(ppd)
type(seq)


# In[13]:


from keras.preprocessing.sequence import pad_sequences
ppd = pad_sequences(seq, padding='post')


# In[14]:


print(type(ppd))
del seq
#split train and test data set
#will have 266457 train data and 66614 test data
#could not use sklearn.model_selection.train_test_split because of memory
X_train, X_test = ppd[:266458], ppd[266458:]
y_train, y_test = labels[:266458], labels[266458:]


# In[15]:


def generator(array, labels, batch_size=10):
    i = 0
    while i <= 266457:
        subarray = array[i: i+batch_size]
        sublabels = labels[i: i+batch_size]
        i = i + batch_size
        yield subarray, sublabels 


# In[16]:


#One hot encode the test data - we pick a small subset of that data to avoid memory error
indices = np.random.randint(0, high=66612, size=300)
xtestsample = X_test[indices]
ytestsample = y_test[indices]


# In[17]:


def create_model():
    model = Sequential()
    model.add(Embedding(21, 3, input_length=12681))
    model.add(Conv1D(128, 6, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Flatten())
    model.add(Dense(33, activation='elu'))
    model.add(Dropout(0.4))
    model.add(Dense(33, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(33, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:



model = create_model()
print(model.summary())
c = 0
Accuracy = []
val_Accuracy = []
Loss = []
val_Loss = []
epochs = 2
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))
fig.tight_layout()
pltcntr = 0
for epoch in range(epochs):
    BatchGenerator = generator(X_train, y_train, batch_size=300)
    print("EPOCH", epoch)
    c = 0
    for X_trainbatch, y_trainbatch in BatchGenerator:
        c, pltcntr = c + 1, pltcntr + 1
        print(" BATCH %d" % c)
        history = model.train_on_batch(X_trainbatch, y_trainbatch)
        scores = model.evaluate(xtestsample, ytestsample, verbose=0)
        print("Loss =", history[0], "Accuracy =", history[1], "Val_Accuracy =", scores[1])
        Loss.append(history[0])
        val_Loss.append(scores[0])
        Accuracy.append(history[1])
        val_Accuracy.append(scores[1])
    print("=======================================================================")


    
#axs[0].scatter(pltcntr, history[0], color='red', s=1)
axs[0].plot(Loss, color='red', linewidth=0.1)
axs[0].plot(val_Loss, color='blue', linewidth=1)
axs[0].set_title("Loss")
axs[0].legend(['loss', 'val_loss'])
axs[0].set_xlabel('batch-epoch')
axs[0].set_ylabel('loss')

#axs[1].scatter(pltcntr, history[1], color='blue', s=1)
axs[1].plot(Accuracy, color='red', linewidth=0.1)
axs[1].plot(val_Accuracy, color='blue', linewidth=1)
axs[1].set_title("Accuracy")
axs[1].legend(['acc', 'val_accuracy'])
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('Accuracy')
plt.show()


# In[36]:


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))
fig.tight_layout()
axs[0].plot(Loss, color='red', linewidth=0.1)
axs[0].plot(val_Loss, color='blue', linewidth=1)
axs[0].set_title("Loss")
axs[0].legend(['loss', 'val_loss'])
axs[0].set_xlabel('batch-epoch')
axs[0].set_ylabel('loss')

#axs[1].scatter(pltcntr, history[1], color='blue', s=1)
axs[1].plot(Accuracy, color='red', linewidth=0.1)
axs[1].plot(val_Accuracy, color='blue', linewidth=1)
axs[1].set_title("Accuracy")
axs[1].legend(['acc', 'val_accuracy'])
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('Accuracy')
plt.show()


# In[34]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[ ]:




