from optimistic_initial_values import run_experiment as oiv
from UCB1_optimistic_initial_values import run_experiment as uvb1
from comparing_epsilons import run_experiment as eps
from bayesian import run_experiment as bay
from bayesian import run_experiment_decaying_epsilon as decay_eps
import matplotlib.pyplot as plt

exp1 = oiv(1, 2, 3, 10, 100000)
exp2 = uvb1(1, 2, 3, 10, 100000)
exp3 = eps(1, 2, 3, 0.05, 100000)
exp4 = decay_eps(1, 2, 3, 100000)
exp5 = bay(1, 2, 3, 100000)

plt.plot(exp1, label="Opt")
plt.plot(exp2, label="UVB1")
plt.plot(exp3, label="Eps")
plt.plot(exp4, label="EpsDec")
plt.plot(exp5, label="Bay")

plt.xscale("log")
plt.legend()
plt.show()

plt.plot(exp1, label="Opt")
plt.plot(exp2, label="UVB1")
plt.plot(exp3, label="Eps")
plt.plot(exp4, label="EpsDec")
plt.plot(exp5, label="Bay")

plt.legend()
plt.show()
