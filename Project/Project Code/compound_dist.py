import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

SIZE = 10_000

Y = stats.norm.rvs(size=SIZE)
Z = stats.gamma.rvs(1, 1, size=SIZE)
X = [stats.norm.rvs(m, s, size=1)[0] for m, s in zip(Y, Z)]
print(X)
pat, bins = np.histogram(X, bins=30) 
plt.hist(X, bins=bins, label="Compound", alpha=0.5)
plt.hist(stats.norm.rvs(0,1, size=SIZE),bins=bins, label="Std Normal", alpha=0.5)
plt.legend()
plt.show()