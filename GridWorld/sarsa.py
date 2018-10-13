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
    