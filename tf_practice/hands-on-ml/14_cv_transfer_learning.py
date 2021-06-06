# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation,Dropout
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

test_size = 3000
train_size = 5000

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
print(X_train_full.shape)

X_train = X_train_full[:train_size, :, :] / 255.
y_train = y_train_full[:train_size]
X_test = X_test[:test_size, :, :] / 255.
y_test = y_test[:test_size]
print(X_train.shape)

from tensorflow.keras.applications.resnet import ResNet50
pretrained_model = ResNet50(
    weights = 'imagenet'
    ,include_top = False
    ,input_shape = (32,32,3)
)
print(pretrained_model.summary())
for l in pretrained_model.layers:
    print(l)
    l.trainable = False

x = pretrained_model.output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10,activation='softmax')(x)
# print(pretrained_model.summary())

model = tf.keras.models.Model( pretrained_model.input ,x )

model.compile(
    loss = 'sparse_categorical_crossentropy'
    ,metrics=['accuracy']
    ,optimizer='Adam'
)
history = model.fit(
    X_train,
    y_train,
    epochs=3,
)
