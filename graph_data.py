import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def graph_model(model, max):
    x = np.arange(1, max)
    y = model.predict(x, verbose=False)
    plt.plot(x,y, 'r')
    plt.xlabel("body mass (grams)")
    plt.ylabel("predicted brain mass (grams)")
    plt.show()

if __name__ == "__main__":
    model_resource = '2x100-hidden-layers-model'
    model = load_model(model_resource)
    graph_model(model, 300000)
