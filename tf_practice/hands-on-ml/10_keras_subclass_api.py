### import data
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

### import data
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
X_train_full =  X_train_full[:5000,:,:]/255.
y_train_full =  y_train_full[:5000]
X_test = X_test[:3000,:,:]/ 255.
y_test = y_test[:3000]

### subclass api
# tofix
class WideAndDeepModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = tf.keras.layers.Input(shape=(28,28))
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_deep = tf.keras.layers.Dense(128,activation = 'relu')
        self.hidden_softmax = tf.keras.layers.Dense(10,activation='softmax')
#
    def call(self,inputs):
        input_flatten = self.flatten(inputs)
        layer1 = self.hidden_deep(input_flatten)
        layer2 = self.hidden_deep(layer1)
        output = self.hidden_softmax(layer2)
        return output
model3 = WideAndDeepModel()
model3.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"]
)
print('model3')
#
# model3.fit(
#     X_train_full.reshape(-1,28,28,)
#     , y_train_full
#     ,validation_data=(X_test, y_test)
#     ,epochs = 3
# )