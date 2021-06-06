import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
X_train_full =  X_train_full[:5000,:,:]/255.
y_train_full =  y_train_full[:5000]
X_test = X_test[:3000,:,:]/ 255.
y_test = y_test[:3000]


### sequential api
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28))
    ,tf.keras.layers.Dense(256,activation  = 'relu')
    ,tf.keras.layers.Dense(10,activation= 'softmax')
])
print('sequential model')
print(model.summary())
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)
model.fit(
    X_train_full, y_train_full
    ,validation_data=(X_test, y_test)
    ,epochs = 3
)