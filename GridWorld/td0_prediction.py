import numpy as np
import matplotlib.pyplot as plt
from grid import negative_grid
from policy_iteration import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALPHA = 0.1 # learning rate
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps=0.1):
    if np.random.random() < (1-eps):
        return a
    
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(policy, grid):
    s = (2, 0)
    grid.set_state(s) # start in bottom left corner
    states_and_rewards = [(s, 0)] # starting position empty

    while not grid.game_over():
        a = random_action(policy[s])
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    return states_and_rewards




        

if __name__ == '__main__':
    grid = negative_grid()

    # Prediction problem, so fixed policy
    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }

    print('\nRewards:')
    print_values(grid.rewards, grid)

    print('\nPolicy:')
    print_policy(policy, grid)

    V = {}
    for s in grid.all_states():
        V[s] = 0

    for i in range(1000):
        states_and_rewards = play_game(policy, grid)
        for j in range(len(states_and_rewards) - 1):
            s1, _ = states_and_rewards[j]
            s2, r = states_and_rewards[j+1]

            # Update V(s) AS we experience the episode
            V[s1] = V[s1] + ALPHA * (r + GAMMA*V[s2] - V[s1])

    print('\nValues:')
    print_values(V, grid)
