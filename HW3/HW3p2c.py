import numpy as np
from scipy.optimize import newton
from matplotlib import pyplot as plt

PARAMS = {
    "groups": 4,
     "recovery_rate": 3,
     "contact_rate": 0.5,
     "susceptibility_probabilities": [1,2,3,4],
     "groupwise_population": [1000, 1000, 1000, 1000]
     }

class TimeDomain:

    def __init__(self, start_time: float, end_time: float, n_steps: int):
        self.start_time = start_time
        self.end_time = end_time 
        self.n_steps = n_steps
        self.spacing = (self.end_time - self.start_time) / (self.n_steps)
        self.array = np.linspace(self.start_time, self.end_time, self.n_steps+1)


class EulerSolver():
    def solve(
            self,
            time_domain: TimeDomain,
            model,
            initial_conditions: np.array,
    ):
        solution_buffer = []
        solution_buffer.append(initial_conditions)
        for t in time_domain.array:
            y_new = solution_buffer[-1] + time_domain.spacing * (model.rhs(solution_buffer[-1], t))
            solution_buffer.append(y_new)
        return np.array(solution_buffer)

# Multigroup infection model
class MultigroupInfection():
    def __init__(
        self, 
        *, 
        groups: int,
        recovery_rate: float,
        contact_rate: float,
        susceptibility_probabilities: list,
        groupwise_population: list,
    ) -> None:
        self.groups=groups
        self.recovery_rate=recovery_rate
        self.contact_rate=contact_rate
        self.susceptibility_probabilities=np.array(susceptibility_probabilities)
        self.groupwise_population=groupwise_population
        self.ones = np.ones(self.groups)

    def rhs(self, state: np.array, time:float):
        ret = np.empty_like(state)
        infections = state[:self.groups]
        recoveries = state[self.groups:]
        new_infections = self.contact_rate * self.susceptibility_probabilities * (1-infections-recoveries) * (self.ones@infections) - (self.recovery_rate * infections)
        new_recoveries = self.recovery_rate * infections
        ret[:self.groups] = new_infections
        ret[self.groups:] = new_recoveries
        return ret
    
    def get_epidemic_initial_conditions(self, rate):
        ret = self.groups * [rate]
        ret = ret + self.groups * [0]
        return ret

mgi_model = MultigroupInfection(**PARAMS)
time = TimeDomain(0, 10, 1_000)
solver = EulerSolver()

# Run multigroup infection simulation
initial_conditions = mgi_model.get_epidemic_initial_conditions(0.1)
mgi_solution = solver.solve(time, mgi_model, np.array(initial_conditions))

# Settings
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 15

titles = ["Susceptable", "Infected", "Recovered"]
fig, ax = plt.subplots(1, 1)
labels = ["$i_1$", "$i_2$", "$i_3$", "$i_4$"]
for group in range(mgi_model.groups):
    ax.plot(time.array, mgi_solution[:, group][:-1], 'b', alpha=1/(mgi_model.groups - group)**2, label=labels[group])

ax.set_xlabel("Time, $t$ [s]")
ax.legend()

plt.show()
