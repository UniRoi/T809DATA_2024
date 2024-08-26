# Author: 
# Date:
# Project: 
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


def _square_error(y, y_hat):
    pass


def _plot_mean_square_error():
    pass


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass


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

    pass
