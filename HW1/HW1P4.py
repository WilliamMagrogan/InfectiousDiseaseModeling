import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

# model parameters
parameter_pairs = [
    (1, 0.5)
]
t_bounds = (0, 50)

# system of differential equations
def f(y_vec, *args):
    beta, gamma = args
    s, i, r = y_vec
    snext = -beta*i*s 
    inext = beta*s*i-gamma*i
    rnext = gamma*i
    return np.array([snext, inext, rnext])


# Solver parameters
n_steps = int(1e3)

def euler_solve(fn, init_conds, t_bounds, ns, args):
    ys = [np.array(init_conds)]
    dt = (t_bounds[1]-t_bounds[0])/ns
    for t in range(n_steps):
        ynext = ys[-1]+dt*fn(ys[-1], *args)
        ys.append(ynext)
    return np.array(ys)

def main():
    plt.rcParams["axes.prop_cycle"]=plt.cycler('color', ['#0000FF', '#FF0000', '#000000'])
    fig, axs = plt.subplots(1, 1)
    for iax, ax in enumerate([axs]):
        sol = euler_solve(f, [0.4, 1e-6, 0.6-1e-6], t_bounds, n_steps, parameter_pairs[iax])
        lines = ax.semilogy(np.linspace(t_bounds[0], t_bounds[1], n_steps), int(1e6)*sol[:-1, :])
        print(sol[-1, :])
        ax.legend(lines, ["S", "I", "R"])
    for ax in [axs]:
        ax.set(xlabel=f"time", ylabel=f"logarithem of population")
    plt.show()

if __name__=="__main__":
    main()