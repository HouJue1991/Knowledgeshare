import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM,TimeDistributed,Dense
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

print(generate_time_series(10000, 100))

n_steps = 50
train_size=7000
test_size = 2000

series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:train_size, :n_steps], series[:train_size, -1]
X_valid, y_valid = series[train_size:train_size+test_size, :n_steps], series[train_size:train_size+test_size, -1]
print(X_train.shape)
model = tf.keras.Sequential([
    LSTM(64,return_sequences=True,input_shape = (None , 1))
    ,LSTM(64,return_sequences=True)
    ,TimeDistributed(Dense(10))
])
model.compile(
    loss="mse"
    ,optimizer="adam"
)
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_valid, y_valid)
)