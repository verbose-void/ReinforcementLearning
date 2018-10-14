import numpy as np
import matplotlib.pyplot as plt
from grid import negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_eval_start import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = negative_grid()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    # No policy initialization, we'll derive our policy from most recent Q

    # Init Q(s, a)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    
    # Init adaptive learning rates
    update_counts = {} # debugging purposes
    update_counts_sa = {} # adaptive learning rate
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    # main loop, repeat until convergence
    t = 1.0 # epsilon greedy
    deltas = []
    for it in range(10000):
        # Decaying Epsilon (Hyper-Parameter, can be changed based on testing to fit your needs)
        if it % 100 == 0:
            t += 10e-3
        if it % 2000 == 0:
            print("it:", it)

        # Starting State
        s = (2, 0)
        grid.set_state(s)

        a, _ = max_dict(Q[s])
        delta = 0
        while not grid.game_over():
            a = random_action(a, eps=0.5/t) # epsilon greedy
            # uniform random action also works, but it would be slower as it doesn't
            # try to go to the end goal

            r = grid.move(a)
            s2 = grid.current_state()

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]

            a2, max_q_s2a2 = max_dict(Q[s2])
            # THIS IS WHERE Q-LEARNING DIFFERS FROM SARSA
            # The Q(s, a) is calculated, even though a isn't guarenteed to be
            # the next action taken. (as the random_action function is used
            # at the beginning of the containing loop.)
            Q[s][a] = old_qsa + alpha*(r + GAMMA*max_q_s2a2 - old_qsa)
            delta = max(delta, np.abs(old_qsa - Q[s][a]))

            # log how often Q(s) has been updated as well
            update_counts[s] = update_counts.get(s,0) + 1

            s = s2
            a = a2

        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    # determine the policy* from Q*
    # find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what's the portion of time we spend updating each part of Q?
    print("\nUpdate Counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)