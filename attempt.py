from __future__ import print_function

#Credit goes to ||Source|| for providing the resources necessary for me to
#make this project generalizable to a different dataset.

import pandas as pd

#Importing stuff

import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
from keras.preprocessing.text import Tokenizer
import re

#Github only allows 25mb
#df = pd.read_csv('dialogueText_301.csv', nrows = 200000)
#df.to_csv('dialogueText_301_small.csv')

df = pd.read_csv('dialogueText_301_small.csv')

text = df['text'].dropna()

print(text.head())


tokenizer = Tokenizer()

#I cant do more than this probably
text_sub = text[1:29999]

#text_sub.to_csv('text_ubuntu_dialogue_corpus_small')

#Parses sentences into words and makes a dictionary 
# So like 'apple', 1 or 'orange', 10
tokenizer.fit_on_texts(text_sub)

#Returns a unique dictionary 
word_idx = tokenizer.word_index

#Sequences will be the words that were converted to numbers 
#according to word_idx's dictionary
sequences = tokenizer.texts_to_sequences(text_sub)

#We're going to pad the matrices of numbers that used to be words.
# Every value from the 'text' column of the csv file is not the same length
# Some of those values are length like 100, aka 100 words
# we need the max length of this so that we can make every sequence (what used to be a sentence
# or a row in the dataset['text']) and make it the same length, we can put 0s where there
# are no words
max_length = max([len(seq) for seq in sequences])

#This is where we pad it
# say one row had "what's", "up"
# row two had "hi"
# say row 1 was encoded to 1,2
# and row 2 was 12
# Then the padded matrix would be [1,2][12,0]
padded_sequences = pad_sequences(sequences, maxlen=max_length)

#All but the last column
X = padded_sequences[:, :-1]
#Include only last column
y = padded_sequences[:, -1]


#y = np.eye(len(word_idx) + 1)[y]

#add 1 to account for the prediction y_hat
num_words = len(tokenizer.word_index) + 1

#1 hot encodes
y = np.eye(num_words)[y]

# define model architecture
model = Sequential()

#Embedding layer for predictions
model.add(Embedding(input_dim=len(word_idx) + 1, output_dim=50, input_length=max_length-1))

#Account for sequential dependencies and words that depend on another
model.add(LSTM(64))

#Hidden layer outputting probability distribution
model.add(Dense(len(word_idx) + 1, activation='softmax'))


# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=10, verbose=1)

model.save('my_model.h5')


import pickle

with open('tokenizer.pickle', 'wb') as e:
    pickle.dump(tokenizer, e, protocol=pickle.HIGHEST_PROTOCOL)

with open('word_idx.pkl', 'wb') as f:
    pickle.dump(word_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

