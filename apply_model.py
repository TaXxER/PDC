import os
import numpy as np
from keras.models import model_from_yaml
from scipy.stats import hmean

path = os.path.abspath("preprocessed_training_logs/training_log_10.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

model = model_from_yaml(open('model_architectures/process_10.yaml').read())
model.load_weights('model_weights/process_10.h5')

# cut the text in semi-redundant sequences of maxlen characters
step = 1
sentences = []
next_chars = []
lines = text.splitlines()
lines2 = text.splitlines()
lines = map(lambda x: '{'+x+'}',lines)
maxlen = max(map(lambda x: len(x),lines))
lines = map(lambda x: x.ljust(maxlen),lines)

chars = map(lambda x : set(x),lines)
chars = set().union(*chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

for line in lines:
    for i in range(0, len(line), step):
        sentences.append(line[0: i])
        next_chars.append(line[i])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#print('Vectorization...')
#X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
#y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
#for i, sentence in enumerate(sentences):
#    for t, char in enumerate(sentence):
#        X[i, t, char_indices[char]] = 1
#    y[i, char_indices[next_chars[i]]] = 1
#
#X2 = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
#for i, sentence in enumerate(sentences2):
#    for t, char in enumerate(sentence):
#        X2[i, t, char_indices[char]] = 1

def calculate_likelihood(sentence):
    p=1
    v = list()
    for t, char in enumerate(sentence):
        x = np.zeros((1, t+1, len(chars)))
        t2 = len(sentence)-(t+1)
        x2 = np.zeros((1, t2+1, len(chars)))
        x[0, 0, char_indices['{']] = 1.
        x2[0, t2, char_indices['}']] = 1.
        for i in range(t):
            x[0, i+1, char_indices[sentence[i]]] = 1.
        for i in range(t2):
            x2[0, i, char_indices[sentence[t+i+1]]] = 1.
        preds = model.predict([x,x2], verbose=0)[0]
        #print x
        #print preds
        #print char
        #print "char: %s" % preds[char_indices[char]]
        p=p*preds[char_indices[char]]
        v.append(preds[char_indices[char]])
        #print "agg:  %s." % p
        #print
    x = np.zeros((1, len(sentence)+1, len(chars)))
    t2 = len(sentence)-(len(sentence)+1)
    x2 = np.zeros((1, 1, len(chars)))
    x2[0, 0, char_indices[' ']] = 1.
    x[0, 0, char_indices['{']] = 1.
    for i in range(len(sentence)):
        x[0, i+1, char_indices[sentence[i]]] = 1.
    preds = model.predict([x,x2], verbose=0)[0]
    #print x
    #print preds
    #print 'end'
    #print "char: %s" % preds[char_indices['}']]
    p=p*preds[char_indices['}']]
    v.append(preds[char_indices['}']])
    #print "agg:  %s" % p
    #print "avg:  %s" % (sum(v)/len(v))
    #print "min:  %s" % min(v)
    #print
    #return hmean(v)
    try:
        return min(v)
    except ValueError:
        return 0
print 'calculating minima train set'

minima = map(lambda x: calculate_likelihood(x), lines2)

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
density = gaussian_kde(minima)
xs = np.linspace(0,max(minima))
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.figure(1)
plt.subplot(211)
plt.plot(xs,density(xs))

print 'calculating minima test set'

path2 = os.path.abspath("preprocessed_test_logs/test_log_may_10.txt")#get_file('encoded_log_wil.txt', origin="E:/Git/sequence_modelling/encoded_log_wil.txt")
print '1'
text2 = open(path2).read().lower()
print '2'
lines3 = text2.splitlines()
print '3'
minima2 = map(lambda x: calculate_likelihood(x), lines3)
print 'plotting minima test set'


arr = np.array(minima2)
arr.argsort()[-10:][::-1]