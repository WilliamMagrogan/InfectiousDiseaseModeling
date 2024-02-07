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

def i_exact(t, paramters, ic):
    beta, gamma = paramters
    s0, i0 = ic
    iinf = 1-gamma/beta
    V = iinf/i0 - 1
    return iinf / (1+V*np.exp(-(beta-gamma)*t))

def abs_error(dt):
    soln = euler_solve(f, [0.99, 0.01], t_bounds, int(25//dt), parameter_pairs[0])
    infections = soln[:-1,1]
    ts = np.linspace(0, 25, int(25//dt))
    exacts_infections = i_exact(ts, parameter_pairs[0], [0.99, 0.01])
    return np.max(np.abs(exacts_infections - infections))


def main():

    dts = [2**-ii for ii in range(-1, 7)]
    errs = [abs_error(dt) for dt in dts]
    fit = np.polyfit(np.log(dts), np.log(errs), deg=1)
    print(fit)
    plt.loglog(dts, errs, "ro")
    plt.xlabel("Log $\Delta t$")
    plt.ylabel("Log absolute error")
    plt.show()

    

if __name__=="__main__":
    main()