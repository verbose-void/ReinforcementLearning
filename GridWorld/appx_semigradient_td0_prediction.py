import numpy as np
import matplotlib.pyplot as plt
from grid import standard_grid
from iterative_policy_evaluation import standard_grid, print_policy, print_values
from td0_prediction import play_game, SMALL_ENOUGH, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS


# NOTE: This is only policy eval not optimization.

class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self, s):
        # convert state to X (our one-hot-encoder)
        # may not be good features
        return np.array([s[0]-1, s[1]-1.5, s[0]*s[1]-3, 1])

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.s2x(s)


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

    model = Model()
    deltas = []

    k = 1.0
    for it in range(20000):
        if it % 10 == 0:
            k += 0.01
        if it % 2000 == 0:
            print(it)

        alpha = ALPHA/k  # deaying learning rate
        delta = 0

        states_and_rewards = play_game(policy, grid)

        for t in range(len(states_and_rewards)-1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]

            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA*model.predict(s2)

            model.theta += alpha*(target - model.predict(s))*model.grad(s)
            delta = max(delta, np.abs(old_theta - model.theta).sum())

        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0  # terminal state or can't get to it

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)
