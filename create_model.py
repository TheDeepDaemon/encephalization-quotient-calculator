import keras
from keras import Sequential
from keras.layers import Dense, Input
from load_data import load_data
import numpy as np
from keras.models import load_model
from graph_data import graph_model
from load_data import load_data
import matplotlib.pyplot as plt


# the usual prediction function
def old_prediction_function(body_mass_grams):
    return 0.12 * (body_mass_grams**(2/3))


def train_model():
    model = Sequential()
    model.add(Input(shape=1))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation=None))

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse')

    # load real data
    x, y = load_data()

    # train it based on the original function
    dummy_x = np.linspace(0, 100_000_000, num=10)
    dummy_y = old_prediction_function(dummy_x)
    model.fit(dummy_x, dummy_y, epochs=1_000_000)

    # then train it using the real data
    model.fit(x, y, epochs=10_000)

    model.save('models/2x50-hidden-layers-model')


if __name__ == "__main__":
    train_model()
    model_resource = 'models/2x50-hidden-layers-model'
    model = load_model(model_resource)
    graph_model(model, 300_000, load_data())
