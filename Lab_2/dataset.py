from typing import Tuple

import numpy as np
from matplotlib import pyplot as plot
import pandas as pd

from ct import X_COORD, Y_COORD, CLASSES, get_ds_path


def create_class(mean: float, st_dev: float, size: int) -> np.array:
    return np.random.normal(loc=mean,
                            scale=st_dev,
                            size=(size, 2))


def show_classes(ds: pd.DataFrame):
    plot.plot(ds[ds[CLASSES] == 0][X_COORD], ds[ds[CLASSES] == 0][Y_COORD], 'bo')
    plot.plot(ds[ds[CLASSES] == 1][X_COORD], ds[ds[CLASSES] == 1][Y_COORD], 'ro')
    plot.show()


def create_class_in_square(min_pos: float, max_pos: float, size: int):
    return np.stack([np.random.random(size)*max_pos + min_pos, np.random.random(size)*max_pos + min_pos], axis=1)


def gen_and_save_datasets():
    def gen_dataset(args1: Tuple, args2: Tuple) -> Tuple[np.array, np.array]:
        cl1 = create_class(*args1)
        cl2 = create_class(*args2)
        return cl1, cl2

    def save_dataset(cl1: np.array, cl2: np.array, size1: int, size2: int, name: str):
        ds = np.vstack([cl1, cl2])
        df = pd.DataFrame({X_COORD: ds[:, 0],
                           Y_COORD: ds[:, 1],
                           CLASSES: np.concatenate([np.ones(size1), np.zeros(size2)])})
        df.to_csv(get_ds_path(name))

    def gen_and_save_dataset(args1: Tuple, args2: Tuple, name: str):
        cl1, cl2 = gen_dataset(args1, args2)
        save_dataset(cl1, cl2, args1[-1], args2[-1], name)

    gen_and_save_dataset((12, 2, 100), (0, 2, 100), "ds1")
    gen_and_save_dataset((12, 2, 1000), (0, 2, 1000), "ds2")
    gen_and_save_dataset((12, 2, 300), (0, 2, 50), "ds3")
    gen_and_save_dataset((6, 2, 300), (0, 2, 50), "ds4")
    gen_and_save_dataset((3.5, 4, 300), (0, 2, 50), "ds5")
    gen_and_save_dataset((12, 2, 100), (0, 2, 100), "ds1")

    ds6_cl1 = create_class_in_square(1, 2, 100)
    ds6_cl2 = create_class(0, 2, 100)
    save_dataset(ds6_cl1, ds6_cl2, 100, 100, "ds6")


def show_ds(name):
    df = pd.read_csv(get_ds_path(name))
    show_classes(df)


def show_datasets():
    show_ds("ds1")
    show_ds("ds2")
    show_ds("ds3")
    show_ds("ds4")
    show_ds("ds5")
    show_ds("ds6")


if __name__ == '__main__':
    show_datasets()
