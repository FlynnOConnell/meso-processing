import tensorflow
from pathlib import Path
import keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the model
    modelpath = Path().home() / 'Dropbox' / 'RBO' / 'LBM-sample' / 'Code' / 'Segmentation_Executable' / 'cnn_model.h5'
    model = tensorflow.keras.models.load_model(modelpath)

    # Plot the model's architecture
    tensorflow.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Display the plot
    img = plt.imread('model_plot.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
