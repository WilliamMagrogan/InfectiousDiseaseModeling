import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

r0s = [0.9, 1.0, 1.1, 2]

def curry(R0):
    return lambda r: 1-np.exp(- R0 * r)

fig, axs = plt.subplots(2, 2)

xs = np.linspace(0, 0.5, 100)

for iax, ax in enumerate(axs.flat):
    f = curry(r0s[iax])
    ax.plot(xs, xs, 'k')
    ax.plot(xs, f(xs), 'r')
    ax.set_title(f"Intersection when R0={r0s[iax]}")

    sol = fsolve(lambda r: r- f(r), [0, 1])
    ax.scatter(sol, sol, c='b')

    print(sol)

for ax in axs.flat:
    ax.set(xlabel="$r_\infty$")

plt.show()