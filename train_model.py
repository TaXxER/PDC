'''
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import os

path = os.path.abspath("E:/event_logs/contest/training_log_2.txt")#get_file('encoded_log_wil.txt', origin="E:/Git/sequence_modelling/encoded_log_wil.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

# cut the text in semi-redundant sequences of maxlen characters
step = 1
hidden_units = 150
sentences = []
sentences2 = []
next_chars = []
lines = text.splitlines()
lines = map(lambda x: '{'+x+'}',lines)
maxlen = max(map(lambda x: len(x),lines))


chars = map(lambda x : set(x),lines)
chars = set().union(*chars)
print('total chars:', len(chars))
lines = map(lambda x: x.ljust(maxlen),lines)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

for line in lines:
    for i in range(0, len(line), step):
        sentences.append(line[0: i])
        if (i+1) <= len(line):
            sentences2.append(line[i+1:len(line)])
        next_chars.append(line[i])
        
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        if char==' ':
            continue
        X[i, t, char_indices[char]] = 1
    if next_chars[i]==' ':
        continue
    y[i, char_indices[next_chars[i]]] = 1

X2 = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences2):
    for t, char in enumerate(sentence):
        if char==' ':
            continue
        X2[i, t, char_indices[char]] = 1

# build model
def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks
    
left = Sequential()
left.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars))))
left.add(Dropout(0.4))

right = Sequential()
right.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars)), go_backwards=True))
right.add(Dropout(0.4))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

left = Sequential()
left.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars))))
left.add(Dropout(0.4))

right = Sequential()
right.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars)), go_backwards=True))
right.add(Dropout(0.4))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

left = Sequential()
left.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars))))
left.add(Dropout(0.4))

right = Sequential()
right.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars)), go_backwards=True))
right.add(Dropout(0.4))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

left = Sequential()
left.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars))))
left.add(Dropout(0.4))

right = Sequential()
right.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars)), go_backwards=True))
right.add(Dropout(0.4))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

left = Sequential()
left.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars))))
left.add(Dropout(0.4))

right = Sequential()
right.add(LSTM(hidden_units, init='he_normal', return_sequences=False, input_shape=(maxlen, len(chars)), go_backwards=True))
right.add(Dropout(0.4))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Train...")
model.fit([X,X2], y, batch_size=maxlen, nb_epoch=1)
