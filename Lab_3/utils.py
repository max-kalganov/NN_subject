from keras.datasets import mnist
from matplotlib import pyplot as plt


def get_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def display_picture(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()