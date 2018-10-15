import numpy as np
import matplotlib.pyplot as plt
from grid import negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_eval_start import max_dict
from sarsa import random_action, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS

SA2IDX = {}
IDX = 0


class Model():
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def sa2x(self, s, a):
        return np.array([
            # Not my mess, easy clean up but I'm lazy.
            s[0] - 1 if a == 'U' else 0,
            s[1] - 1.5 if a == 'U' else 0,
            (s[0]*s[1] - 3)/3 if a == 'U' else 0,
            (s[0]*s[0] - 2)/2 if a == 'U' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
            1 if a == 'U' else 0,
            s[0] - 1 if a == 'D' else 0,
            s[1] - 1.5 if a == 'D' else 0,
            (s[0]*s[1] - 3)/3 if a == 'D' else 0,
            (s[0]*s[0] - 2)/2 if a == 'D' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
            1 if a == 'D' else 0,
            s[0] - 1 if a == 'L' else 0,
            s[1] - 1.5 if a == 'L' else 0,
            (s[0]*s[1] - 3)/3 if a == 'L' else 0,
            (s[0]*s[0] - 2)/2 if a == 'L' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
            1 if a == 'L' else 0,
            s[0] - 1 if a == 'R' else 0,
            s[1] - 1.5 if a == 'R' else 0,
            (s[0]*s[1] - 3)/3 if a == 'R' else 0,
            (s[0]*s[0] - 2)/2 if a == 'R' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
            1 if a == 'R' else 0,
            1
        ])

    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)


# Turn Q predictions into a dictionary given some state s
def getQs(model, s):
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs


if __name__ == '__main__':
    grid = negative_grid()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    model = Model()

    # two different time values because
    # we want epsilon & alpha to learn at
    # different rates. These are hyper-parameters
    t = 1.0
    t2 = 1.0
    deltas = []
    for it in range(20000):
        if it % 100 == 0:
            t += 10e-3
            t2 += 0.01
        if it % 1000 == 0:
            print(it)
        alpha = ALPHA / t2

        # instead of generating an episode, we will play
        # an episode within this loop, as TD(0) is fully
        # online.
        s = (2, 0)  # start
        grid.set_state(s)

        # get Q(s) so we can choose the first action
        Qs = getQs(model, s)

        a = max_dict(Qs)[0]
        # epsilon-greedy & windy grid world
        a = random_action(a, eps=0.5/t)
        delta = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            # debugging purposes
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha * \
                    (r - model.predict(s, a)) * model.grad(s, a)
            else:
                # not terminal

                # get next optimal action
                Qs2 = getQs(model, s2)
                a2 = max_dict(Qs2)[0]
                a2 = random_action(a2, eps=0.5/t)

                # update Q(s,a) as we experience episode
                # (fully online)
                model.theta += alpha * \
                    (r + GAMMA*model.predict(s2, a2) -
                     model.predict(s, a))*model.grad(s, a)

                s = s2
                a = a2

            delta = max(delta, np.abs(model.theta - old_theta).sum())

        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    V = {}
    policy = {}

    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            max_Qs = max_dict(getQs(model, s))
            V[s] = max_Qs[1]
            policy[s] = max_Qs[0]
        else:
            V[s] = 0

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)
