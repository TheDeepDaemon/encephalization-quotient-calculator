import matplotlib.pyplot as plt
import numpy as np

def graph_model(model, max, data=None):
    x = np.arange(1, max)
    y = model.predict(x, verbose=False)
    plt.plot(x, y, 'r')
    plt.xscale('linear')
    plt.yscale('linear')
    if data is not None:
        plt.plot(data[0], data[1], 'bo')
    plt.xlabel("body mass (grams)")
    plt.ylabel("predicted brain mass (grams)")
    plt.show()

def graph_data(x, y):
    plt.plot(x, y, 'b')
    plt.xlabel("body mass (grams)")
    plt.ylabel("brain mass (grams)")
    plt.show()
