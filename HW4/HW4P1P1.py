import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy import stats

UPPER_DATA_LIMIT = 12

all_data = np.genfromtxt("all_weeks.csv", delimiter=",")
all_data = all_data[2:, :]

def estimator(data):
    xdata = data[:, 0]
    ydata = data[:, 1]
    log_ydata = np.log(ydata)
    slope, intercept, r, p, se = stats.linregress(xdata, log_ydata)
    slope_ci = [slope-1.96*se, slope+1.96*se]
    return intercept, slope, slope_ci



inter, sl, sl_ci = estimator(all_data[:UPPER_DATA_LIMIT])

plt.plot(all_data)
plt.show()
print(np.mean(all_data[-100:, 1]))

fig, axs = plt.subplots(ncols=2, nrows=1, sharex=True)
axs[0].plot(all_data[:UPPER_DATA_LIMIT, 0], all_data[:UPPER_DATA_LIMIT, 1])
axs[0].set_title("True Space")

axs[1].semilogy(all_data[:UPPER_DATA_LIMIT, 0], all_data[:UPPER_DATA_LIMIT, 1], label="Data")
axs[1].semilogy(all_data[:UPPER_DATA_LIMIT, 0], np.exp(inter +sl*all_data[:UPPER_DATA_LIMIT, 0]), label="Estimate")
axs[1].semilogy(all_data[:UPPER_DATA_LIMIT, 0], np.exp(inter+sl_ci[0]*all_data[:UPPER_DATA_LIMIT, 0]), label=r"95%-ile")
axs[1].semilogy(all_data[:UPPER_DATA_LIMIT, 0], np.exp(inter+sl_ci[1]*all_data[:UPPER_DATA_LIMIT, 0]), label=r"5%-ile")
axs[1].legend()
axs[1].set_title("Semi-Log")

print(sl, sl_ci)

plt.show()