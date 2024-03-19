import numpy as np
from scipy.stats import nbinom
from matplotlib import pyplot as plt

class BranchingProcessNegativeBinomial():
    def __init__(
            self,
            *,
            mean: float,
            dispersion: int,
            max_generations: int,
            ):
        self.mean=mean
        self.dispersion=dispersion
        self.max_generations=max_generations
    
    def simulate_run(self):
        generations = 1
        infections_in_current_generation = 1
        cumulative_infections = 1

        var = self.mean + (self.mean**2) / self.dispersion
        p = self.mean / var
        n = self.mean**2 / (var - self.mean)

        for ii in range(self.max_generations):
            new_infections = np.sum(nbinom.rvs(n=n, p=p, size=infections_in_current_generation))
            if new_infections == 0:
                return (
                    generations,
                    infections_in_current_generation,
                    cumulative_infections,
                )
            infections_in_current_generation = new_infections
            cumulative_infections += new_infections
            generations += 1

        return (-1, infections_in_current_generation, cumulative_infections)

R0=3
ks = [0.1, 0.5, 1.0, 5.0]
MAX_GENERATIONS = 5
SIM_RUNS = 10_000

PARAMS = {
    "mean": R0,
    "dispersion": ks[0],
    "max_generations": MAX_GENERATIONS,
}

finite_time_rates = []


# Extra credit portion
bpnb = BranchingProcessNegativeBinomial(**PARAMS)
FINITE_TIME_OUTBREAK_REQUIRED = 100_000
outbreak_count = 0
sizes = []
while outbreak_count<=FINITE_TIME_OUTBREAK_REQUIRED:
    res = bpnb.simulate_run()
    if res[0] == -1:
        continue
    else:
        sizes.append(res[2])
        outbreak_count +=1


plt.hist(sizes, log=True)
plt.xscale('log')
plt.xlabel("Outbreak size")
plt.ylabel("Counts")
plt.title("Finite Time Outbreak Empirical Distribution")
plt.show()

for k in ks:
    PARAMS["dispersion"] = k
    bpnb = BranchingProcessNegativeBinomial(**PARAMS)
    reses = []
    for ii in range(SIM_RUNS):
        res = bpnb.simulate_run()
        reses.append(res)

    reses = np.array(reses)
    generations = reses[:, 0]
    cumulative = reses[:, 2]

    finite_time_count = len(generations[generations!=-1])
    finite_time_rates.append(finite_time_count/SIM_RUNS)

    fig, axs = plt.subplots(1,2)

    axs[0].hist(generations[generations!=-1])
    axs[1].hist(cumulative[generations!=-1])

    axs[0].set_xlabel("Generations")
    axs[0].set_ylabel("Counts")

    axs[1].set_xlabel("Size of Outbreak")
    axs[1].set_ylabel("Counts")
    plt.show()

estimates = [1 - x for x in finite_time_rates]
print(estimates)
