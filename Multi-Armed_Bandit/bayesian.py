import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import Bandit as EpsBandit


class Bandit:
    def __init__(self, m):
        self.m = m
        # params for mu - prior is N(0,1)
        self.m0 = 0
        self.lambda0 = 1
        self.sum_x = 0
        self.tau = 1

    def pull(self):
        return np.random.randn()+self.m

    def sample(self):
        # generates a sample from a gaussian
        return np.random.randn() / np.sqrt(self.lambda0) + self.m0

    def update(self, x):
        self.lambda0 += 1
        self.sum_x += x
        self.m0 = self.tau*self.sum_x / self.lambda0


def run_experiment_decaying_epsilon(m1, m2, m3, N):
    bandits = [EpsBandit(m1), EpsBandit(m2), EpsBandit(m3)]
    data = np.empty(N)

    for i in xrange(N):

        # decaying greedy-epsilon
        p = np.random.random()
        if p < 1.0/(i+1):
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale("log")
    # plt.show()

    return cumulative_average


def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in xrange(N):

        j = np.argmax([b.sample() for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale("log")
    # plt.show()

    return cumulative_average


if __name__ == "__main__":
    e1 = run_experiment_decaying_epsilon(1, 2, 3, 5000)
    e2 = run_experiment_decaying_epsilon(1, 2, 3, 5000)
    e3 = run_experiment_decaying_epsilon(1, 2, 3, 5000)

    e4 = run_experiment(1, 2, 3, 5000)
    e5 = run_experiment(1, 2, 3, 5000)
    e6 = run_experiment(1, 2, 3, 5000)

    # logarithmic
    plt.plot(e1, label="E1-eps")
    plt.plot(e2, label="E2-eps")
    plt.plot(e3, label="E3-eps")

    plt.plot(e4, label="E4-bay")
    plt.plot(e5, label="E5-bay")
    plt.plot(e6, label="E6-bay")

    plt.legend()
    plt.xscale("log")
    plt.show()

    # linear
    plt.plot(e1, label="E1-eps")
    plt.plot(e2, label="E2-eps")
    plt.plot(e3, label="E3-eps")

    plt.plot(e4, label="E4-bay")
    plt.plot(e5, label="E5-bay")
    plt.plot(e6, label="E6-bay")

    plt.legend()
    plt.show()
