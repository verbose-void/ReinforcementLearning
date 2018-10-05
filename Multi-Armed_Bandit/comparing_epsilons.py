import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m):
        self.mean = 0
        self.N = 0
        self.m = m

    def pull(self):
        return np.random.randn()+self.m

    def update(self, x):
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean+(1.0/self.N)*x


def runExperiment(m1, m2, m3, eps, N):
    # eps or epsilon determines the percentage that the algorithm will explore rather than exploit. (trying new slot machines rather than exploiting the best one)

    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        # epsilon-greedy
        p = np.random.randn()

        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x

    cumulative_average = np.cumsum(data) / np.arange(N) + 1
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale("log")
    # plt.show()

    return cumulative_average


if __name__ == "__main__":
    e1 = runExperiment(1, 2, 3, .1, 500)
    e2 = runExperiment(1, 2, 3, .05, 500)
    e3 = runExperiment(1, 2, 3, .01, 500)

    # logarithmic
    plt.plot(e1, label="E1")
    plt.plot(e2, label="E2")
    plt.plot(e3, label="E3")
    plt.legend()
    plt.xscale("log")
    plt.show()

    # linear
    plt.plot(e1, label="E1")
    plt.plot(e2, label="E2")
    plt.plot(e3, label="E3")
    plt.legend()
    plt.show()
