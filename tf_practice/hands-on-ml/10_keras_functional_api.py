### import data
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

### functional api
input_ = tf.keras.layers.Input(shape = (28,28))
flatten = tf.keras.layers.Flatten()(input_)
hidden_deep1 = tf.keras.layers.Dense(128,activation = 'relu')(flatten)
hidden_deep2 = tf.keras.layers.Dense(128,activation = 'relu')(hidden_deep1)
hidden1_wide1 = tf.keras.layers.Dense(256,activation = 'relu')(flatten)
concate1 = tf.keras.layers.concatenate([hidden1_wide1,hidden_deep2])
output_ = tf.keras.layers.Dense(10,activation = 'softmax')(concate1)
model = tf.keras.Model(
    inputs = input_
    ,outputs = output_
)
print('deep wide model using functional_api ')
print(model.summary())
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.fit(
    X_train_full, y_train_full
    ,validation_data=(X_test, y_test)
    ,epochs = 3
)
