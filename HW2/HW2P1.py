import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

# model parameters
parameter_pairs = [
    (3, 2)
]

t_bounds = (0, 25)

# system of differential equations
def f(y_vec, *args):
    beta, gamma = args
    s, i = y_vec
    snext = -beta*i*s +gamma*i
    inext = beta*s*i-gamma*i
    return np.array([snext, inext])


# Solver parameters
n_steps = int(1e3)

def euler_solve(fn, init_conds, t_bounds, ns, args):
    ys = [np.array(init_conds)]
    dt = (t_bounds[1]-t_bounds[0])/ns
    for t in range(ns):
        ynext = ys[-1]+dt*fn(ys[-1], *args)
        ys.append(ynext)
    return np.array(ys)

resolutions = [25//2,25//1,25*2]

def i_exact(t, paramters, ic):
    beta, gamma = paramters
    s0, i0 = ic
    iinf = 1-gamma/beta
    V = iinf/i0 - 1
    return iinf / (1+V*np.exp(-(beta-gamma)*t))

def main():
    plt.rcParams["axes.prop_cycle"]=plt.cycler('color', ['#0000FF', '#FF0000', '#000000'])
    fig, axs = plt.subplots(1, 3)
    for iax, ax in enumerate(axs.flat):
        sol = euler_solve(f, [0.99, 0.01], t_bounds, resolutions[iax], parameter_pairs[0])
        simline, = ax.plot(np.linspace(t_bounds[0], t_bounds[1], resolutions[iax]), sol[:-1, 1],c="r")
        exactline, = ax.plot(np.linspace(t_bounds[0],t_bounds[1],1000), i_exact(np.linspace(t_bounds[0],t_bounds[1],1000),parameter_pairs[0],[0.99, 0.01]), "k--")
        #print(sol[-1, :])
        ax.legend([simline, exactline], ["Forward Euler", "Analytical"])
    for ax in axs.flat:
        ax.set(xlabel=f"time", ylabel=f"population fraction")
        ax.set_ylim(0, 0.5)
    plt.show()

if __name__=="__main__":
    main()