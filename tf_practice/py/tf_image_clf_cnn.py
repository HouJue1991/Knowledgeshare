import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import os

MODEL_PATH = './model/'

###load data
(train_x,train_y) , (test_x , test_y) = cifar10.load_data()
print(train_x.shape,test_x.shape)

# train_x = train_x[:5000]
# train_y= train_y[:5000]
# test_x= test_x[:3000]
# test_y= test_y[:3000]


### preprocess data
train_gen = ImageDataGenerator(
    rescale=1./255.
    # ,horizontal_flip=True
    # ,vertical_flip=True
    # ,rotation_range=40
    # ,width_shift_range=0.2
    # ,height_shift_range=0.2
    # ,fill_mode='nearest'
)

test_gen = ImageDataGenerator(
    rescale=1./255.
    # ,horizontal_flip=True
    # ,vertical_flip=True
    # ,rotation_range=40
    # ,width_shift_range=0.2
    # ,height_shift_range=0.2
    # ,fill_mode='nearest'
)
train_generator = train_gen.flow(
    train_x
    ,train_y
    ,batch_size = 128
)
test_generator = test_gen.flow(
    test_x
    ,test_y
    ,batch_size = 128
)

### design the model
model  = tf.keras.Sequential()
model.add(Conv2D(filters = 64,kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 128,kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 256,kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(
    optimizer='Adam'
    ,loss = 'sparse_categorical_crossentropy'
    ,metrics = ['accuracy' ]
)
### train model


model.fit(
    train_generator
    ,epochs = 20
    ,validation_data=test_generator
)

model.save(MODEL_PATH+'image_classification.h5')

