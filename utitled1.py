import pickle
import numpy as np
import random
import keras
from keras.preprocessing.text import Tokenizer

def add_space(string):
    return ' '.join(string)

with open("data.pkl", 'rb') as f:
    rawProt = pickle.load(f)

prots = np.reshape(rawProt, 400000)

#I use a for instead of map() since I dont wanna mess with the dtype
for i in range(400000):
    prots[i] = add_space(prots[i])


t = Tokenizer()

t.fit_on_texts(prots)

print(t.word_counts)
print('#########################################################')
print(t.document_count)
print('#########################################################')
print(t.word_index)
print(t.word_docs)