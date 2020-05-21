from collections import Callable

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

    def test_x(x) -> bool:
        return x[x.sum(1) < 1].shape[0] == 0

    def test_y(y) -> bool:
        sums = y.sum(0)
        return np.abs(np.mean(sums) - sums).max() <= 1000

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = preproc_and_expand_dataset(x_train, y_train, x_test, y_test)

    reshaped_x_train = reshape_x(x_train)
    reshaped_x_test = reshape_x(x_test)
    one_hot_y_train = one_hot_encoding(y_train)
    one_hot_y_test = one_hot_encoding(y_test)

    assert test_x(reshaped_x_train), f"error in x train"
    assert test_x(reshaped_x_test), f"error in x test"
    assert test_y(one_hot_y_train), f"error in y train"
    assert test_y(one_hot_y_test), f"error in y test"

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

    def gaussian_with_params(image):
        return gaussian(image,
                        sigma=(1.5, 1.5),
                        truncate=3.5,
                        multichannel=False)

    def apply(x: np.array, y: np.array, func: Callable, sample_size: int = 100):
        full_x = None
        full_y = None
        for i in range(10):
            x_selected = x[y == i]
            x_indices = np.random.choice(x_selected.shape[0], sample_size)
            x_selected = x_selected[x_indices]
            x_applied = None
            for j in range(x_selected.shape[0]):
                x_applied = concat_x(x_applied,
                                     func(x_selected[j, :, :])*255,
                                     reshape_new=True)

            full_x = concat_x(full_x, x_applied, False)
            full_y = concat_y(full_y, sample_size, i)

        return np.vstack([x, full_x]), np.concatenate([y, full_y])

    def add_clear(x: np.array, y: np.array, sample_size: int = 1000, border: int = 50):
        clear_x = None
        clear_y = None
        for i in range(10):
            x_selected = x[y == i]
            x_indices = np.random.choice(x_selected.shape[0], sample_size)
            x_selected = x_selected[x_indices]

            x_selected[x_selected <= border] = 0
            x_selected[x_selected > border] = 255

            clear_x = concat_x(clear_x, x_selected, False)
            clear_y = concat_y(clear_y, sample_size, i)

        return np.vstack([x, clear_x]), np.concatenate([y, clear_y])

    def shift(image):
        def get_width_height(height: bool):
            colored_lines = (image < 50).sum(int(height))
            colored_positions = np.where(colored_lines > 0)[0]
            if colored_positions.shape[0] <= 0:
                print("wrong image")
                display_picture(image)
                return None, None, None
            return colored_positions[-1] - colored_positions[0], colored_positions[0], colored_positions[-1]

        width, start_w, end_w = get_width_height(height=False)
        height, start_h, end_h = get_width_height(height=True)
        if width is None or height is None:
            return image

        new_image = np.ones((28, 28)) * 255
        start_x = np.random.randint(0, 28-width)
        start_y = np.random.randint(0, 28-height)
        new_image[start_x:start_x+width+1, start_y:start_y+height+1] = image[start_w:end_w+1, start_h:end_h+1]
        return new_image

    x_train, x_test = 255 - x_train, 255 - x_test
    x_train[x_train < 255] = 0
    x_test[x_test < 255] = 0
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)

    #x_train, y_train = add_clear(x_train, y_train, sample_size=int((x_train.shape[0]/10) // 4), border=50)
    #x_test, y_test = add_clear(x_test, y_test, sample_size=100, border=150)

    # x_train, y_train = add_clear(x_train, y_train, sample_size=int((x_train.shape[0]/10) // 4), border=5)
    # x_test, y_test = add_clear(x_test, y_test, sample_size=100, border=5)

    # x_train, y_train = apply(x_train, y_train, gaussian_with_params, sample_size=int((x_train.shape[0]/10) // 4))
    # x_test, y_test = apply(x_test, y_test, gaussian_with_params, sample_size=100)

    x_train, y_train = apply(x_train, y_train, shift, sample_size=int((x_train.shape[0] / 10) // 2))
    x_test, y_test = apply(x_test, y_test, shift, sample_size=100)



    x_train /= 255
    x_test /= 255

    # full_x = None
    # full_y = None
    # size = np.min([np.sum(y_train == i) for i in range(10)])
    # for i in range(10):
    #     x_selected = x_train[y_train == i]
    #     x_indices = np.random.choice(x_selected.shape[0], size)
    #     full_x = concat_x(full_x, x_selected[x_indices], False)
    #     full_y = concat_y(full_y, size, i)
    #
    # x_train = full_x
    # y_train = full_y
    error_ex_train = x_train[np.sum(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]), axis=1) > 750]
    error_ex_test = x_test[np.sum(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]), axis=1) > 750]
    if len(error_ex_train) > 0:
        print("error in train")
    if len(error_ex_test) > 0:
        print("error in train")

    return (x_train, y_train), (x_test, y_test)


def display_picture(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()


def display_sample(x: np.array, y: np.array, sample_size: int):
    for i in range(10):
        x_selected = x[y == i]
        x_indices = np.random.choice(x_selected.shape[0], sample_size)
        x_selected = x_selected[x_indices]
        for pict in x_selected[:]:
            display_picture(pict.reshape(28, 28))


if __name__ == '__main__':
    get_dataset()
