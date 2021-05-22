import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense

X = np.array(range(30))
y = X * 2 + 10
print(X)
print(y)

model = tf.keras.Sequential()
model.add(Dense(1,input_shape = [1]))

model.compile(
    optimizer=tf.optimizers.Adam(lr = 0.1)
    ,loss = 'mse'
    ,metrics=['mae','mse']
)
print(model.summary())
model.fit(X,y,epochs = 1000)

print(model.predict([3]))
