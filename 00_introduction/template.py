import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True


def normal(x: np.ndarray, sigma: np.float64, mu: np.float64) -> np.ndarray:
    # Part 1.1
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-((x-mu)**2)/ (2*sigma**2))

def plot_normal(sigma: np.float64, mu:np.float64, x_start: np.float64, x_end: np.float64):
    # Part 1.2

    x_range = np.linspace(x_start, x_end, 500)
    p = normal(x_range, sigma, mu)
    plt.plot(x_range, p, label=fr'$\mu$ {mu}, $\sigma$ {sigma}')


def plot_three_normals():
    # Part 1.2
    plt.clf()
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)

    plt.title("Compare normal distributions")
    plt.legend(loc='upper left')
    plt.savefig("1_2_1.png")
    plt.show()


def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    prob = 0
    for i in range(len(sigmas)):
        prob += (weights[i]/np.sqrt(2*np.pi*sigmas[i]**2)) * np.exp(-((x-mus[i])**2)/ (2*sigmas[i]**2))

    return prob

def plot_mixture(sigmas: list, mus: list, weights: list, x_start: np.float64, x_end: np.float64):
    x_range = np.linspace(x_start, x_end, 500)
    p = normal_mixture(x_range, sigmas, mus, weights)
    plt.plot(x_range, p, label=fr'$\mu$ {mus}, $\sigma$ {sigmas}, $\phi$ {weights}')

def compare_components_and_mixture():
    # Part 2.2
    plt.clf()

    plot_normal(0.5, 0, -5, 5)
    plot_mixture([0.5], [0], [1/3], -5, 5)
    
    plot_normal(1.5, -0.5, -5, 5)
    plot_mixture([1.5], [-0.5], [1/3], -5, 5)

    plot_normal(0.25, 1.5, -5, 5)
    plot_mixture([0.25], [1.5], [1/3], -5, 5)

    plt.title("Compare components and mixtures")
    plt.legend(loc='upper left')
    plt.savefig("2_2_1.png")
    plt.show()
    return 4

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    return 5

def _plot_mixture_and_samples():
    # Part 3.2
    return 6

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    
    # print(normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1]))
    # compare_components_and_mixture()
    plot_three_normals()