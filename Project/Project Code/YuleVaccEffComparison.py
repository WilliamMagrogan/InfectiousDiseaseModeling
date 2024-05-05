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
                    unprotected_vaccinated = stats.binom.rvs(number_of_vaccinations, 1-self.vaccination_effectiveness, size=1)[0]
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

nb_mean = 2
nb_variance = 100
nb_r = nb_mean**2 / (nb_variance - nb_mean)
nb_p = nb_mean/nb_variance
yule_rho = 2
beta_alpha = 20
beta_beta = 3
vaccination_effectiveness = .8
vaccination_rate = .2

contact_distribution = stats.yulesimon(yule_rho)
infectiousness_distribution = stats.beta(beta_alpha, beta_beta)

# Heatmap of r0 against infectiousness
generation_buffer = np.zeros((RUN_BATCH_SIZE,6))
ves = np.linspace(0, 0.8, 10)
buff = np.empty_like(ves)
yule_roots = []
yule_r0 = []
nb_roots = []
nb_r0 = []
yule_finite_outbreak_sizes = []
nb_finite_outbreak_sizes = []
for ive, ve in enumerate(ves):
    model_parameters = {
    "contact_distribution":contact_distribution,
    "infectiousness_distribution":stats.beta(beta_alpha, beta_beta),
    "vaccination_effectiveness": ve,
    "vaccination_rate": vaccination_rate,
    "vaccination_type": "leaky",
    "max_iterations": 5
    }

    bp = CoumpoundBranchingProcess(**model_parameters)

    cumulative_emperical_data = []
    yfos = []
    for ii in range(RUN_BATCH_SIZE):
        generational_data, emperical_data = bp.run()
        generation_buffer[ii, :] = generational_data
        cumulative_emperical_data += emperical_data
        if 0 in generational_data:
            yfos.append(np.sum(generational_data))
    unique, counts = np.unique(cumulative_emperical_data, return_counts=True)
    N = len(cumulative_emperical_data)
    freqs = counts/N
    obj = lambda y: np.sum(freqs*y**unique) - y 
    sol = optimize.root_scalar(obj, bracket=[0.01,0.99])
    root = sol.root
    yule_roots.append(root)
    yule_r0.append(np.sum(cumulative_emperical_data)/N)
    yule_finite_outbreak_sizes.append(len(yfos)/RUN_BATCH_SIZE)

    model_parameters = {
    "contact_distribution":contact_distribution,
    "infectiousness_distribution":stats.beta(beta_alpha, beta_beta),
    "vaccination_effectiveness":ve,
    "vaccination_rate": vaccination_rate,
    "vaccination_type": "aon",
    "max_iterations": 5
    }

    bp = CoumpoundBranchingProcess(**model_parameters)

    cumulative_emperical_data = []
    nbfos = []
    for ii in range(RUN_BATCH_SIZE):
        generational_data, emperical_data = bp.run()
        generation_buffer[ii, :] = generational_data
        cumulative_emperical_data += emperical_data
        if 0 in generational_data:
            nbfos.append(np.sum(generational_data))
    unique, counts = np.unique(cumulative_emperical_data, return_counts=True)
    N = len(cumulative_emperical_data)
    freqs = counts/N
    obj = lambda y: np.sum(freqs*y**unique) - y 
    sol = optimize.root_scalar(obj, bracket=[0.01,0.99])
    root = sol.root
    nb_roots.append(root)
    nb_r0.append(np.sum(cumulative_emperical_data)/N)
    nb_finite_outbreak_sizes.append(len(nbfos)/RUN_BATCH_SIZE)



plt.plot(ves, yule_roots, 'ro', label="Yule Leaky")
# plt.plot(alphas, yule_r0, 'r-', label="Yule")
plt.plot(ves, yule_finite_outbreak_sizes, 'r-', label="Yule Leaky Simulations")
plt.plot(ves, nb_roots, 'ko', label="Yule All-or-Nothing")
# plt.plot(alphas, nb_r0, 'k-', label="NB")
plt.plot(ves, nb_finite_outbreak_sizes, 'k-', label="Yule All-or-Nothing Simulations")
plt.title(r"Minor Outbreak Probability agains Vaccine Efficacy$")
plt.xlabel(r"$v$, Vaccination Efficacy")
plt.ylabel("Finite Outbreak Probability")
plt.legend()
plt.show()



