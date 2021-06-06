# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , BatchNormalization,Activation
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib as mpl
import matplotlib.pyplot as plt

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(X_train_full.shape)
X_train = X_train_full[:5000, :, :] / 255.
y_train = y_train_full[:5000]
X_test = X_test[:3000, :, :] / 255.
y_test = y_test[:3000]
print(X_train.shape)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(X_train.shape)
### basic cnn
model = tf.keras.Sequential([
    Conv2D(filters=64, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu')
    , MaxPooling2D()
    , Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    , MaxPooling2D()
    , Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    , MaxPooling2D()
    , Flatten()
    , Dense(units=64, activation='relu')
    , Dense(10, activation='softmax')
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

