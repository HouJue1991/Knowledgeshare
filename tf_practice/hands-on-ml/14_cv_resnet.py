# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
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


### ResNet-34
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.main_layers = [
            Conv2D(filters=filters, kernel_size=(3, 3), strides=strides)
            , BatchNormalization()
        ]
        if stride > 1:
            self.skip_layers = [
                Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)
                , BatchNormalization()
            ]

        def call(self, inputs):
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            return tf.keras.activations.relu(Z + skip_Z)


model = keras.models.Sequential([
    Conv2D(filters= 64 , kernel_size=(4,4), strides=2, input_shape=[28, 28, 1])
    , BatchNormalization()
    , Activation('relu')
    , MaxPooling2D()
]
)
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

