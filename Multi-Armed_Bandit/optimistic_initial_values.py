import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m, upper_limit):
        # optimistic initial value
        self.mean = upper_limit
        self.N = 0
        self.m = m

    def pull(self):
        return np.random.randn()+self.m

    def update(self, x):
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean+(1.0/self.N)*x


def run_experiment(m1, m2, m3, upper_limit, N):
    bandits = [Bandit(m1, upper_limit), Bandit(
        m2, upper_limit), Bandit(m3, upper_limit)]
    data = np.empty(N)

    for i in xrange(N):
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


if __name__ == "__main__":
    e1 = run_experiment(1, 2, 3, 10, 5000)
    e2 = run_experiment(1, 2, 3, 10, 5000)
    e3 = run_experiment(1, 2, 3, 10, 5000)

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
