# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import RNN,LSTM, BatchNormalization, Activation,Dropout
import os

# txt_data_path = './data/shakespeare.txt'
# with open(txt_data_path) as f:
#     txt =  f.read()
# print(txt[:100])
#
# tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
# tokenizer.fit_on_texts(txt)
# print(tokenizer.index_word)
times = 10000
results = np.array([[0, 0, 1, 2], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]])
n = 100
# print(results)
print(results)
print(results[:,3])
print(np.argmax(results[:,3]))