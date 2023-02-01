import tensorflow as tf
import keras
from keras import Sequential, layers
from keras.layers import Dense, Input
from load_data import load_data
import numpy as np

model = Sequential()
model.add(Input(shape=1))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation=None))

model.summary()

model.compile(optimizer='rmsprop', loss='mse')

data = load_data()

x = data[0]
y = data[1]

x = np.reshape(x, (x.shape[0], 1))
y = np.reshape(y, (y.shape[0], 1))

model.fit(x, y, epochs=20000)

print(model.predict([100, 1000, 10000, 100000]))

model.save('2x100-hidden-layers-model')

