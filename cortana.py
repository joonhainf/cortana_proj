from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import numpy as np
import nltk
import pickle

with open('tokenizer.pickle', 'rb') as e:
    tokenizer = pickle.load(e)

with open('word_idx.pkl', 'rb') as f:
    word_idx = pickle.load(f)

idx_word = {v: k for k, v in word_idx.items()}
model = load_model('my_model.h5')
max_length = model.layers[0].input_shape[1]

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/cortana', methods=['POST'])
def cortana():

    user_input = request.json['user_input']
    input_seq = tokenizer.texts_to_sequences([user_input])[0]
    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='pre')
    pred = model.predict(input_seq)
    pred_idx = np.argmax(pred)
    pred_word = idx_word[pred_idx]
    #return pred_word

    return jsonify({'bot_response': pred_word})

if __name__ == '__main__':
    app.run()