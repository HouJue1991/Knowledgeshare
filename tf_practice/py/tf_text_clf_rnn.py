import subprocess
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,LSTM,Bidirectional
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

### load data and preprocess
max_len = 32
trunc_type='post'
padding_type='post'

(train_x,train_y) , (test_x , test_y) = imdb.load_data()
word2idx_dict = imdb.get_word_index()
# idx2word_dict = {idx:word for word,idx in word2idx_dict.items()}
# print(train_x.shape,test_x.shape)
# def idxs2sentence(idxs):
#     return [idx2word_dict[idx] for idx in idxs]
# print(train_x[1])
# print(idxs2sentence(train_x[1]))
vocab_size = len(word2idx_dict)

train_x = pad_sequences(
    train_x
    ,maxlen = max_len
    ,truncating=trunc_type
    ,padding =padding_type
)
test_x = pad_sequences(
    test_x
    , maxlen = max_len
    , truncating=trunc_type
    , padding=padding_type
)
print(train_x[1])

### design the model
model  = tf.keras.Sequential()
model.add(Embedding(input_dim=vocab_size ,output_dim= 10,input_length=max_len ))
model.add(Bidirectional(LSTM(128,return_sequences = True )))
model.add(Bidirectional(LSTM(128)))
# model.add(Dense(128 ,activation = 'relu'))
model.add(Dense(1 ,activation = 'sigmoid'))

model.compile(
    optimizer='Adam'
    ,loss = 'binary_crossentropy'
    ,metrics = ['accuracy' ]
)
print(model.summary())

### train mode
model.fit(
    train_x,train_y
    ,epochs = 5
    ,validation_data=(test_x , test_y)
)
#
model_path = './model/'
model.save(model_path+'text_classification.h5')
#
