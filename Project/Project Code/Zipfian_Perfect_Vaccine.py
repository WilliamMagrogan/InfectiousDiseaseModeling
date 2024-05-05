import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, optimize, integrate
from scipy.special import beta, binom, zeta

class CoumpoundBranchingProcess():
    def __init__(
            self, 
            *, 
            contact_distribution,
            infectiousness_distribution,
            vaccination_effectiveness,
            vaccination_rate,
            vaccination_type,
            max_iterations = 5,
    ) -> None:
        self.contact_distribution = contact_distribution
        self.infectiouness_distribution = infectiousness_distribution
        self.vaccination_effectiveness = vaccination_effectiveness
        self.vaccination_rate = vaccination_rate
        self.vaccination_type = vaccination_type
        self.max_iterations = max_iterations

    def run(self):
        infections_in_generation = [1]
        empirical_secondary_infections = []
        for _ in range(self.max_iterations):
            cumulative_secondary_infections = 0
            for _ in range(infections_in_generation[-1]):
                primary_infection_probability = self.infectiouness_distribution.rvs(size=1)[0]
                primary_infection_contacts = self.contact_distribution.rvs(size=1)[0]
                number_of_vaccinations = stats.binom.rvs(primary_infection_contacts, self.vaccination_rate, size=1)[0]
                number_of_susceptibles = primary_infection_contacts - number_of_vaccinations

                if self.vaccination_type=="aon":
                    unprotected_vaccinated = stats.binom.rvs(number_of_vaccinations, self.vaccination_effectiveness, size=1)[0]
                    secondary_infections = stats.binom.rvs(number_of_susceptibles + unprotected_vaccinated, primary_infection_probability, size=1)[0] 
                elif self.vaccination_type=="leaky":
                    infected_susceptibles = stats.binom.rvs(number_of_susceptibles, primary_infection_probability, size=1)[0] 
                    infected_vaccinated = stats.binom.rvs(number_of_vaccinations, primary_infection_probability * (1-self.vaccination_effectiveness), size=1)[0] 
                    secondary_infections = infected_susceptibles + infected_vaccinated
                else:
                    raise ValueError("Invalid vaccination type.")

                empirical_secondary_infections.append(secondary_infections)
                cumulative_secondary_infections += secondary_infections
            infections_in_generation.append(cumulative_secondary_infections)

        return (infections_in_generation, empirical_secondary_infections)
    
    def secondary_infections(self, n:int):
        quad_function = lambda k, l, p: self.contact_distribution.pmf(k) * stats.binom.pmf(self.vaccination_rate, k, l) * self.infectiouness_distribution.pdf(p)
        cumulative = 0
        for k in range(n, 11):
            for l in range(0, k-n+1):
                qf = lambda p: quad_function(k, l, p)
                temp = integrate.quad(qf, 0, 1)
                cumulative += temp[0]
        return cumulative

    def estimate_finite_outbreak_probability(self):
        coeffs = [self.secondary_infections(j) for j in range(10)]
        coeffs[1] = coeffs[1] -1
        roots = np.roots(coeffs[::-1])
        return roots  
    
    def numerical_mean_R0(self, batch_size):
        cum = 0
        for jj in range(batch_size):
            primary_infection_probability = self.infectiouness_distribution.rvs(size=1)[0]
            primary_infection_contacts = self.contact_distribution.rvs(size=1)[0]
            vaccinations = stats.binom.rvs(primary_infection_contacts, self.vaccination_rate, size=1)[0]
            remaining_susceptible = primary_infection_contacts - vaccinations
            secondary_infections = stats.binom.rvs(remaining_susceptible, primary_infection_probability, size=1)[0]
            cum += secondary_infections
        return cum / batch_size

RUN_BATCH_SIZE = 1_000

zipf_a = 3
beta_alpha = 5
beta_beta = 1
vaccination_effectiveness = 0
vaccination_rate = 0

contact_distribution = stats.zipf(zipf_a)
infectiousness_distribution = stats.beta(beta_alpha, beta_beta)

# model_parameters = {
#     "contact_distribution":contact_distribution,
#     "infectiousness_distribution":infectiousness_distribution,
#     "vaccination_effectiveness":vaccination_effectiveness,
#     "vaccination_rate": vaccination_rate,
#     "vaccination_type": "aon",
#     "max_iterations": 8
# }

# buffer = np.zeros((RUN_BATCH_SIZE, model_parameters["max_iterations"]+1))

# branching_process = CoumpoundBranchingProcess(**model_parameters)
# cumulative_emperical_data = []

# for ii in range(RUN_BATCH_SIZE):
#     generational_data, emperical_data = branching_process.run()
#     buffer[ii, :] = generational_data
#     cumulative_emperical_data += emperical_data

# fig, axs = plt.subplots(nrows=3, ncols=3)
# for iax, ax in enumerate(axs.flat):
#     ax.hist(buffer[:, iax], density=True, log=True)
#     ax.set_title(f"Generation {iax}")
#     ax.set_ylabel("Log Frequency")
#     ax.set_xlabel("Infections in Generation")
# plt.show()

# plt.hist(cumulative_emperical_data, cumulative=True, density=True, log=True)
# plt.show()

# Heatmap ofr0 against infectiousness
alphas = range(1, 10)
betas = range(1, 10)
A, B = np.meshgrid(alphas, betas)
buff = np.empty_like(A)
for ia, a in enumerate(alphas):
    for ib, b in enumerate(betas):
        
        model_parameters = {
        "contact_distribution":contact_distribution,
        "infectiousness_distribution":stats.beta(a, b),
        "vaccination_effectiveness":vaccination_effectiveness,
        "vaccination_rate": vaccination_rate,
        "vaccination_type": "aon",
        "max_iterations": 5
        }
        bp = CoumpoundBranchingProcess(**model_parameters)

        cumulative_emperical_data = []

        for ii in range(RUN_BATCH_SIZE):
            generational_data, emperical_data = bp.run()
            cumulative_emperical_data += emperical_data
        buff[ia, ib] = np.sum(cumulative_emperical_data) / len(cumulative_emperical_data)
plt.pcolormesh(A, B, buff)
plt.colorbar()
plt.xlabel(r"$\alpha$ Parameter")
plt.ylabel(r"$\beta$ Parameter")
plt.title(r"$R_0$ as a function of Infection Distribution Parameters")
plt.show()

# # Function of dispersion parameter
# a_params = np.linspace(2, 4, 5)
# a_buff = np.empty_like(a_params)
# for ia, a in enumerate(np.linspace(2, 4, 5)):
#     zipf_a = a

#     model_parameters = {
#     "contact_distribution":contact_distribution,
#     "infectiousness_distribution":infectiousness_distribution,
#     "vaccination_effectiveness":vaccination_effectiveness,
#     "vaccination_rate": vaccination_rate,
#     "vaccination_type": "aon",
#     "max_iterations": 8
#     }      

#     cumulative_emperical_data = []

#     for ii in range(RUN_BATCH_SIZE):
#         generational_data, emperical_data = branching_process.run()
#         buffer[ii, :] = generational_data
#         cumulative_emperical_data += emperical_data

#     r0_estimate = np.sum(cumulative_emperical_data) / len(cumulative_emperical_data)
#     a_buff[ia] = r0_estimate
# plt.plot(a_params, a_buff)
# plt.show()
