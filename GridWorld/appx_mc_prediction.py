import numpy as np
import matplotlib.pyplot as plt
from grid import standard_grid
from iterative_policy_evaluation import print_values, print_policy

# NOTE: this is policy eval not optimization

from monte_carlo_random import random_action, play_game, GAMMA

LEARNING_RATE = 0.001
SMALL_ENOUGH = 10e-4

if __name__ == '__main__':
    grid = standard_grid()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "U",
        (2, 1): "L",
        (2, 2): "U",
        (2, 3): "L",
    }

    # Init theta
    # Model: V_hat = theta.dot(x)
    theta = np.random.randn(4) / 2
    def s2x(s):
        # x = [row, col, row*col, 1] - 1 for bias term
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

    deltas = []
    t = 1.0
    for it in range(20000):
        # decaying learning rate
        if it % 100 == 0:
            t += 0.01
        alpha = LEARNING_RATE / t

        # generate an episode using pi
        delta = 0
        states_and_returns = play_game(grid, policy) # fixed policy
        seen_states = set()

        for s, G in states_and_returns:
            if s not in states_and_returns:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)

                # grad(V_hat) wrt theta = x
                theta += alpha*(G - V_hat) * x
                delta = max(delta, np.abs(old_theta - theta).sum())
                seen_states.add(s)

        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:
            V[s] = 0

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)