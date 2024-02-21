import numpy as np
from scipy.optimize import newton
from matplotlib import pyplot as plt

PARAMS = {
     "recovery_rate": 1/14,
     "basic_reproduction_rate": 5,
     "vaccine_effectiveness": .8,
     "vaccination_rate": .5,
     "total_population": 300_000,
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

# SIRV-ANM
class SIRVANM():
    def __init__(
            self, 
            *, 
            recovery_rate: float,
            basic_reproduction_rate: float,
            vaccine_effectiveness: float,
            vaccination_rate: float,
            total_population: float,
    ) -> None:
        self.recovery_rate=recovery_rate
        self.basic_reproduction_rate=basic_reproduction_rate
        self.vaccine_effectiveness= vaccine_effectiveness
        self.vaccination_rate=vaccination_rate
        self.total_population=total_population

    def rhs(self, state: np.array, time:float):
        ret = np.empty_like(state)

        S = state[0:1]
        I = state[1:2]
        V0 = state[3:4]

        SN = -self.recovery_rate * self.basic_reproduction_rate * S * I /self.total_population
        IN = self.recovery_rate * self.basic_reproduction_rate * (S + V0) * I /self.total_population - self.recovery_rate * I
        RN = self.recovery_rate * I 
        V0N = -self.recovery_rate * self.basic_reproduction_rate * V0 * I /self.total_population
        VallN = 0

        ret[0:1],ret[1:2],ret[2:3],ret[3:4],ret[4:5]=(SN, IN, RN, V0N, VallN)

        return ret
    
    def get_epidemic_initial_conditions(self):
        S = (self.total_population - 1) * (1 - self.vaccination_rate)
        I = 1
        R = 0
        V0 = (self.total_population - 1) * self.vaccination_rate * (1-self.vaccine_effectiveness)
        Vall = (self.total_population - 1) * self.vaccination_rate - V0
        return np.array([S, I, R, V0, Vall])
    
    @property
    def HIT(self):
        R0 = self.basic_reproduction_rate
        N = self.total_population
        v = self.vaccination_rate
        VE = self.vaccine_effectiveness
        return N/R0 - v*(1-VE)*(N-1)

# SIRV-ANM
class LeakySIRV():
    def __init__(
            self, 
            *, 
            recovery_rate: float,
            basic_reproduction_rate: float,
            vaccine_effectiveness: float,
            vaccination_rate: float,
            total_population: float,
    ) -> None:
        self.recovery_rate=recovery_rate
        self.basic_reproduction_rate=basic_reproduction_rate
        self.vaccine_effectiveness= vaccine_effectiveness
        self.vaccination_rate=vaccination_rate
        self.total_population=total_population

    def rhs(self, state: np.array, time:float):
        ret = np.empty_like(state)

        S = state[0:1]
        I = state[1:2]
        V = state[2:3]

        SN = -self.recovery_rate * self.basic_reproduction_rate * S * I /self.total_population
        IN = self.recovery_rate * self.basic_reproduction_rate * (S + V*(1-self.vaccine_effectiveness)) * I / self.total_population - self.recovery_rate * I
        RN = self.recovery_rate * I 
        VN = -self.recovery_rate * self.basic_reproduction_rate * V * (1-self.vaccine_effectiveness) * I / self.total_population

        ret[0:1],ret[1:2],ret[2:3],ret[3:4]=(SN, IN, RN, VN)

        return ret
    
    def get_epidemic_initial_conditions(self):
        S = (self.total_population - 1) * (1 - self.vaccination_rate)
        I = 1
        R = 0
        V = (self.total_population - 1) * self.vaccination_rate * self.vaccine_effectiveness
        return np.array([S, I, R, V])
    
    @property
    def HIT(self):
        R0 = self.basic_reproduction_rate
        N = self.total_population
        v = self.vaccination_rate
        VE = self.vaccine_effectiveness
        return N/R0 - v*(1-VE)*(N-1)

sirvanm = SIRVANM(**PARAMS)
leakysirv = LeakySIRV(**PARAMS)
time = TimeDomain(0, 700, 7_000)
solver = EulerSolver()

# Run SIRVANM simulation
initial_conditions = sirvanm.get_epidemic_initial_conditions()
anm_solution = solver.solve(time, sirvanm, np.array(initial_conditions))

# Run Leaky SIRV simulation
initial_conditions = leakysirv.get_epidemic_initial_conditions()
leaky_solution = solver.solve(time, leakysirv, np.array(initial_conditions))

# Settings
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 15

M = 3
titles = ["Susceptable", "Infected", "Recovered"]
fig, axs = plt.subplots(1, M)
for iax, ax in enumerate(axs):
    line1 = ax.plot(time.array, anm_solution[:, iax][:-1], 'k', label="All-or-Nothing Model")
    line2 = ax.plot(time.array, leaky_solution[:, iax][:-1], 'r', label="Leaky Model")
    ax.set_xlabel("Time, $t$ [s]")
    ax.set_title(f"{titles[iax]}")
    ax.legend()
plt.show()
