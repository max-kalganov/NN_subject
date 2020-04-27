import numpy as np


def create_class(mean: float, st_dev: float, size: int) -> np.array:
    return np.random.normal(loc=mean,
                            scale=st_dev,
                            size=(size, 2))


def show_classes(ds_cl1: np.array, ds_cl2: np.array):
    plot.plot(ds_cl1[:, 0], ds_cl1[:, 1], 'bo')
    plot.plot(ds_cl2[:, 0], ds_cl2[:, 1], 'ro')
    plot.show()


def create_class_in_square(min_pos: float, max_pos: float, size: int):
    return np.stack([np.random.random(size)*max_pos + min_pos, np.random.random(size)*max_pos + min_pos], axis=1)


if __name__ == '__main__':
    ds1_cl1 = create_class(12, 2, 100)
    ds1_cl2 = create_class(0, 2, 100)

    ds2_cl1 = create_class(12, 2, 1000)
    ds2_cl2 = create_class(0, 2, 1000)

    ds3_cl1 = create_class(12, 2, 300)
    ds3_cl2 = create_class(0, 2, 50)

    ds4_cl1 = create_class(6, 2, 300)
    ds4_cl2 = create_class(0, 2, 50)

    ds5_cl1 = create_class(3.5, 4, 300)
    ds5_cl2 = create_class(0, 2, 50)

    ds6_cl1 = create_class_in_square(1, 2, 100)
    ds6_cl2 = create_class(0, 2, 100)

    from matplotlib import pyplot as plot
    show_classes(ds1_cl1, ds1_cl2)
    show_classes(ds2_cl1, ds2_cl2)
    show_classes(ds3_cl1, ds3_cl2)
    show_classes(ds4_cl1, ds4_cl2)
    show_classes(ds5_cl1, ds5_cl2)
    show_classes(ds6_cl1, ds6_cl2)
