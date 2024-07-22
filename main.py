import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.src.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 name='b')

    def call(self, input):
        return input @ self.w + self.b

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = MyLayer(64)
        self.layer_2 = MyLayer(3)

    def call(self, input):
        x = self.layer_1(input)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.softmax(x)
        return x

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

model = NeuralNetwork()
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

model.fit(x_train, y_train, batch_size=16, epochs=50)
print(model.evaluate(x_test, y_test_cat))
