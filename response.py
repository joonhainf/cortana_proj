#The cortana.py code is better than this one

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import json
import numpy as np
import nltk
import pickle
nltk.download('punkt')


with open('tokenizer.pickle', 'rb') as e:
    tokenizer = pickle.load(e)

with open('word_idx.pkl', 'rb') as f:
    word_idx = pickle.load(f)

idx_word = {v: k for k, v in word_idx.items()}

model = load_model('my_model.h5')

# Embedding layer length (max)

max_length = model.layers[0].input_shape[1]



def cortana(input_text):

    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    print(input_seq)
    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='pre')
    print(input_seq)
    pred = model.predict(input_seq)
    pred_idx = np.argmax(pred)
    print(pred_idx)
    pred_word = idx_word[pred_idx]
    return pred_word


while True:
    
    user_input = input('You: ')
    
    bot_response = cortana(user_input)

    print('Bot:', bot_response)


