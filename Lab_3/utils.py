from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.filters import gaussian
import numpy as np


def one_hot_encoding(y):
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y.reshape(len(y), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def get_dataset():
    def reshape_x(x):
        return x.reshape(x.shape[0], x.shape[1]*x.shape[2])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = preproc_and_expand_dataset(x_train, y_train, x_test, y_test)

    reshaped_x_train = reshape_x(x_train)
    reshaped_x_test = reshape_x(x_test)
    one_hot_y_train = one_hot_encoding(y_train)
    one_hot_y_test = one_hot_encoding(y_test)

    return (reshaped_x_train, one_hot_y_train),\
           (reshaped_x_test, one_hot_y_test)


def preproc_and_expand_dataset(x_train, y_train, x_test, y_test):
    def concat_y(prev, sample_size, i):
        if prev is not None:
            prev = np.concatenate([prev, np.ones(sample_size) * i])
        else:
            prev = np.ones(sample_size) * i
        return prev

    def concat_x(prev, new, reshape_new: bool):
        if reshape_new:
            reshaped_new = new.reshape(1, new.shape[0], new.shape[1])
        else:
            reshaped_new = new
        if prev is not None:
            prev = np.vstack([prev, reshaped_new])
        else:
            prev = reshaped_new
        return prev

    def add_blurred(x: np.array, y: np.array):
        def gaussian_with_params(image):
            return gaussian(image,
                            sigma=(1.5, 1.5),
                            truncate=3.5,
                            multichannel=False)
        blurred_x = None
        blurred_y = None
        sample_size = 100
        for i in range(10):
            x_selected = x[y == i]
            x_indices = np.random.choice(x_selected.shape[0], sample_size)
            x_selected = x_selected[x_indices]
            x_blurred = None
            for j in range(x_selected.shape[0]):
                x_blurred = concat_x(x_blurred,
                                     gaussian_with_params(x_selected[j, :, :]),
                                     reshape_new=True)

            blurred_x = concat_x(blurred_x, x_blurred, False)
            blurred_y = concat_y(blurred_y, sample_size, i)

        return np.vstack([x, blurred_x]), np.concatenate([y, blurred_y])

    def add_clear(x: np.array, y: np.array):
        clear_x = None
        clear_y = None
        sample_size = 1000
        for i in range(10):
            x_selected = x[y == i]
            x_indices = np.random.choice(x_selected.shape[0], sample_size)
            x_selected = x_selected[x_indices]
            x_selected[x_selected <= 50] = 0
            x_selected[x_selected > 50] = 255

            clear_x = concat_x(clear_x, x_selected, False)
            clear_y = concat_y(clear_y, sample_size, i)

        return np.vstack([x, clear_x]), np.concatenate([y, clear_y])

    x_train, x_test = 255 - x_train, 255 - x_test

    x_train, y_train = add_blurred(x_train, y_train)
    x_test, y_test = add_blurred(x_test, y_test)

    x_train, y_train = add_clear(x_train, y_train)
    x_test, y_test = add_clear(x_test, y_test)

    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def display_picture(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()


if __name__ == '__main__':
    get_dataset()
