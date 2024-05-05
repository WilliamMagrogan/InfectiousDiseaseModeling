import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

class CoumpoundBranchingProcess():
    def __init__(
            self, 
            *, 
            contact_distribution,
            infectiousness_distribution,
            vaccination_distribution,
            vaccination_rate,
            max_iterations = 5,
    ) -> None:
        self.contact_distribution = contact_distribution
        self.infectiouness_distribution = infectiousness_distribution
        self.vaccination_distribution = vaccination_distribution
        self.vaccination_rate = vaccination_rate
        self.max_iterations = max_iterations

    def run(self):
        infections_in_generation = [1]
        for ii in range(self.max_iterations):
            cumulative_secondary_infections = 0
            for jj in range(infections_in_generation[-1]):
                primary_infection_probability = self.infectiouness_distribution.rvs(size=1)[0]
                primary_infection_contacts = self.contact_distribution.rvs(size=1)[0]
                vaccinations = stats.binom.rvs(primary_infection_contacts, vaccination_rate, size=1)[0]
                remaining_susceptible = primary_infection_contacts - vaccinations
                secondary_infections = stats.binom.rvs(remaining_susceptible, primary_infection_probability, size=1)[0]
                cumulative_secondary_infections += secondary_infections
            infections_in_generation.append(cumulative_secondary_infections)
        return infections_in_generation

RUN_BATCH_SIZE = 10_000

zipf_a = 3
beta_alpha = 5
beta_beta = 1
vaccination_rate = 0

contact_distribution = stats.zipf(zipf_a)
infectiousness_distribution = stats.beta(beta_alpha, beta_beta)
vaccination_distribution = stats.bernoulli(vaccination_rate)
vaccination_rate = 0

model_parameters = {
    "contact_distribution":contact_distribution,
    "infectiousness_distribution":infectiousness_distribution,
    "vaccination_distribution":vaccination_distribution,
    "vaccination_rate": 0,
    "max_iterations": 5
}

branching_model = CoumpoundBranchingProcess(**model_parameters)
buffer = np.zeros((RUN_BATCH_SIZE, model_parameters["max_iterations"]+1))
for ii in range(RUN_BATCH_SIZE):
    buffer[ii, :] = np.array(branching_model.run())

fig, axs = plt.subplots(nrows=2, ncols=3)
for iax, ax in enumerate(axs.flat):
    ax.hist(buffer[:, iax], density=True, log=True)
    ax.set_title(f"Generation {iax}")
    ax.set_ylabel("Log Frequency")
    ax.set_xlabel("Infections in Generation")
plt.show()


