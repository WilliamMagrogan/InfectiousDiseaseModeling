import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

nb_mean = 4
nb_variance = 5
nb_r = nb_mean**2 / (nb_variance - nb_mean)
nb_p = nb_mean/nb_variance

ps = range(0,10)
pdf = stats.nbinom.pmf(ps, nb_r, nb_p)
plt.title(r"Beta RV: $\alpha=12, \beta=3$")
plt.xlabel(r"$p$")
plt.ylabel(r"Probability $\beta(12, 3)=p$")
plt.plot(ps, pdf, 'ro')
plt.show()