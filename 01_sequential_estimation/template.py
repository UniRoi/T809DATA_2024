# Author: Matthias Reiser
# Date: 26.8. - 28.8.24
# Project: Computer Exercise 1
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    I_k = np.identity(k)

    cov = var**2 * I_k

    return np.random.multivariate_normal(mean, cov, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    new_mu = mu + 1/n*(x-mu)

    return new_mu


def plot_sequence_estimate():
    data = gen_data(100, 2, np.array([0, 0]), 3)
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):

        new_mu = update_sequence_mean(estimates[-1], data[i], i+1)
        estimates.append(new_mu)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')

    plt.legend(loc='upper center')
    plt.savefig("T809DATA_2024/01_sequential_estimation/4_1.png")
    plt.show()


def square_error(y, y_hat):
    return (y - y_hat)**2


def plot_mean_square_error():
    data = gen_data(100, 2, np.array([0, 0]), 3)
    estimates = [np.array([0, 0])]
    sqrt_err_list = [np.array([0, 0])]
    for i in range(data.shape[0]):

        new_mu = update_sequence_mean(estimates[-1], data[i], i+1)
        sqrt_err = square_error(np.array([0, 0]), new_mu)
        estimates.append(new_mu)
        sqrt_err_list.append(sqrt_err)

    sqrt_err_list.pop(0)
    sqrt_mean = np.mean(np.array(sqrt_err_list), axis=1)
    plt.plot(sqrt_mean)
    # plt.plot([e[1] for e in estimates], label='Second dimension')

    # plt.legend(loc='upper center')
    plt.savefig("T809DATA_2024/01_sequential_estimation/5_1.png")
    plt.show()


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    
    # # Section 1
    # np.random.seed(1234)
    # print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    # np.random.seed(1234)
    # print(gen_data(5, 1, np.array([0.5]), 0.5))

    # # Section 2
    # np.random.seed(1234)
    # X = gen_data(300,2,np.array([-1,2]), np.sqrt(4))
    # # print(X.shape[0])

    # scatter_2d_data(X)
    # bar_per_axis(X)
    # plt.show()

    # # Section 3
    # mean = np.mean(X, 0)
    # new_x = gen_data(1, 2, np.array([0, 0]), 1)
    # new_mu = update_sequence_mean(mean, new_x, X.shape[0]+1)
    # print(new_mu)

    # Section 4
    # np.random.seed(1234)
    # plot_sequence_estimate()

    # Section 5
    # plot_mean_square_error()
    pass
