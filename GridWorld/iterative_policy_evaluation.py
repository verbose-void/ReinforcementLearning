from grid import standard_grid
import numpy as np

SMALL_ENOUGH = 10e-4  # convergence threshold


def print_values(V, g):
    print("-----------------------------")
    for i in xrange(g.height):
        temp = "|"
        for j in xrange(g.width):
            v = V.get((i, j), 0)
            if v >= 0:
                temp += " %0.2f " % v + "|"
            else:
                temp += "%0.2f " % v + "|"  # - symbol takes up 1 space
        print(temp)
    print("-----------------------------")


def print_policy(P, g):
    print("-------------")
    for i in xrange(g.height):
        temp = "|"
        for j in xrange(g.width):
            temp += " " + P.get((i, j), " ") + "|"
        print(temp)
    print("-------------")


if __name__ == "__main__":
    grid = standard_grid()
    states = grid.all_states()

    # Uniformly Random Actions
    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0

    # Iterative Policy is the action of building the value function
    while True:
        delta = 0
        for s in states:
            old_v = V[s]

            if s in grid.actions:
                new_v = 0
                p_a = 1.0 / len(grid.actions[s])  # equal action distribution

                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    # Bellman Equation
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                delta = max(delta, np.abs(old_v - V[s]))

        if delta < SMALL_ENOUGH:
            break

    print("\nVals for uniformly random actions:")
    print_values(V, grid)

    # Fixed Policy
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

    V = {}
    for s in states:
        V[s] = 0
    gamma = 0.9

    # Fixed policy is deterministic stepping through values to build the value function.
    while True:
        delta = 0
        for s in states:
            old_v = V[s]
            if s in policy:
                grid.set_state(s)
                r = grid.move(policy[s])
                V[s] = r + gamma * V[grid.current_state()]
                delta = max(delta, np.abs(old_v - V[s]))
        if delta < SMALL_ENOUGH:
            break

    print("\nVals for uniformly fixed policy:")
    print_values(V, grid)
    print_policy(policy, grid)
