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


# In[28]:


np.version.version


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


# In[54]:


print(t.word_index)
print(ppd.dtype, ppd.shape)
def one_hotter(arr):
    hot = np.zeros((12681, 20))
    for i in range(12681):
        if arr[i] != 0:
            hot[i, arr[i]-1] = 1
    return hot

def generator(array, labels, batch_size=12):
     i = 0
    while i <= 333061
        subarray = array[i: i+12]
        sublabels = labels[i: i+12]
        one_hotted = np.apply_along_axis(one_hotter, axis=1, arr=subarray)
        i = i + 10
        yield one_hotted, sublabels 


# In[61]:


ht = np.zeros((12681, 20))
tst = ppd[0:12, :]
a = np.apply_along_axis(one_hotter, axis=1, arr=tst)
print(a[3, 5, :], a.shape)
print(ppd[3, 5])


# In[51]:
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))




# In[ ]:




