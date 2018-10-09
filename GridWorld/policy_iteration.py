import numpy as np
from grid import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")

# this world is deterministic
# all p(s', r| s,a) = 1 or 0

if __name__ == "__main__":
    grid = negative_grid()
    print("\nRewards:")
    print_values(grid.rewards, grid)

    # Random policy generation
    policy = {}
    for pos in grid.actions.keys():
        policy[pos] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print("\nInitial Policy:")
    print_policy(policy, grid)

    # 0 init value function
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # Outer loop forces alternation between iteration and improvement.
    while True:
        # Follow Policy.
        while True:
            delta = 0

            for s in states:
                old_v = V[s]
                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA * V[grid.current_state()]
                delta = max(delta, np.abs(old_v - V[s]))
            if delta < SMALL_ENOUGH:
                break

        converged = True
        for s in states:
            if s in grid.actions:
                old_a = policy[s]
                new_a = None
                best_value = float("-inf")
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    converged = False

        if converged:
            break

    print("\nValues:")
    print_values(V, grid)

    print("\nOptimal Policy:")
    print_policy(policy, grid)
