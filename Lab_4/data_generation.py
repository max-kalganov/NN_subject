from typing import Callable
import numpy as np


def func(y, alpha, beta, t):
    return alpha / np.sqrt(np.abs(y)) + np.sum(np.exp(beta) * (beta**(1/t))) * (beta*alpha/t)


def get_input_timeseries(num_of_timesteps: int) -> np.array:
    """
    :return: result shape = num_of_timesteps * 100 + 1
    """
    alpha = np.random.random(1)*2 + 0.0001
    ksi = np.random.random(num_of_timesteps) - 0.5
    beta = 1 + ksi

    y = np.random.random(1)
    for t in range(1, num_of_timesteps):
        y = np.concatenate([y, func(y[-1], alpha=alpha, beta=beta, t=t)])

    return y


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.plot(get_input_timeseries(50))
    plt.show()
    print()
