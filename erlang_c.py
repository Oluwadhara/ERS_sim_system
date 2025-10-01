#erlang_c.py

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def erlang_c_formula(lmbda, mu, c):
    rho = lmbda / (c * mu)
    if rho >= 1: return 1.0  # unstable system

    def P0():
        summation = sum([(lmbda/mu)**k / factorial(k) for k in range(c)])
        last_term = (lmbda/mu)**c / (factorial(c) * (1 - rho))
        return 1.0 / (summation + last_term)

    P0_val = P0()
    Pc = ( (lmbda/mu)**c / (factorial(c) * (1 - rho)) ) * P0_val
    return Pc

# Parameters
mu = 1/15   # avg service rate (1 per 15 mins)
c = 5       # ambulances
arrival_rates = np.linspace(0.1, 0.9*c*mu, 20)
waiting_probs = [erlang_c_formula(l, mu, c) for l in arrival_rates]

plt.plot(arrival_rates, waiting_probs, marker="o")
plt.xlabel("Arrival rate λ (calls per minute)")
plt.ylabel("Prob. of waiting")
plt.title("Erlang C: Probability of Waiting vs Arrival Rate")
plt.grid(True)
plt.show()
