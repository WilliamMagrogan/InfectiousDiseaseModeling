import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

ps = np.linspace(0, 1, 1_000)
pdf = stats.beta.pdf(ps, 12, 3)
plt.title(r"Beta RV: $\alpha=12, \beta=3$")
plt.xlabel(r"$p$")
plt.ylabel(r"Probability $\beta(12, 3)=p$")
plt.plot(ps, pdf)
plt.show()