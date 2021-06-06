### import data

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, ReLU, LeakyReLU
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
X_train_full = X_train_full[:5000, :, :] / 255.
y_train_full = y_train_full[:5000]
X_test = X_test[:3000, :, :] / 255.
y_test = y_test[:3000]
# train deep neural networks

input = Input(shape=(28, 28))
flatten = Flatten(input_shape=(28, 28))(input)
x1 = Dense(256, activation='relu')(flatten)

x2 = Dense(256, kernel_initializer='he_normal')(flatten)
x2 = LeakyReLU()(x2)

x3 = Dense(256, activation='relu', kernel_initializer='he_normal')(flatten)
x3 = BatchNormalization()(x3)

x4 = Dense(256, activation='relu', kernel_regularizer='l2')(flatten)

concat = tf.keras.layers.concatenate([x1, x2, x3, x4])
output = Dense(10, activation='softmax')(concat)

model = tf.keras.Model(
    inputs=input
    , outputs=output
)
# print(model.summary())
# tf.keras.utils.plot_model(model,'./model/dnn_model.png',show_shapes=True)

# optmizer = tf.keras.optimizers.SGD(lr = 0.001, decay=1e-4)
# optmizer = tf.keras.optimizers.RMSprop(lr = 0.001, decay=1e-4)
# optmizer = tf.keras.optimizers.Adagrad(lr = 0.001, decay=1e-4)
optmizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-4)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optmizer,
    metrics=["accuracy"]
)

from tensorflow.keras.callbacks import LearningRateScheduler


def step_decay(epoch):
    x = 0.001 / float(epoch + 1)
    return x

callback_lr_scheduler = LearningRateScheduler(step_decay)

history = model.fit(
    X_train_full, y_train_full
    , validation_data=(X_test, y_test)
    , epochs=3
    , callbacks=[callback_lr_scheduler]
)
plt.plot(history.epoch, history.history["lr"], "o-")
plt.grid(True)
plt.show()
