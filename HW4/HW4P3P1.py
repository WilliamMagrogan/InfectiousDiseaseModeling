from matplotlib import pyplot as plt
import numpy as np

all_data = np.genfromtxt("HW4_Q3_data.csv", delimiter=",")
neg_ctrl = np.genfromtxt("HW4_Q3_neg.csv", delimiter=",")
pos_ctrl = np.genfromtxt("HW4_Q3_pos.csv", delimiter=",")

fig, ax = plt.subplots()

ax.eventplot([all_data, neg_ctrl, pos_ctrl], colors=['b', 'r', 'k'], orientation="vertical", lineoffsets=2, linewidth=0.75)
plt.show()

def sensitivity(c):
    field_pos = np.sum(pos_ctrl>= c)
    N = len(pos_ctrl)
    return field_pos/N

def specificity(c):
    field_neg = np.sum(neg_ctrl<= c)
    N = len(neg_ctrl)
    return field_neg/N

def raw_prevalance(c):
    field = np.sum(all_data>= c)
    total = len(all_data)
    return field/total

def corr_prevalence(c):
    num = raw_prevalance(c) - (1-specificity(c))
    den = specificity(c)+sensitivity(c)-1
    return num / den

def youden(c):
    return sensitivity(c)+specificity(c) - 1


cs = np.linspace(5, 50, 1_000)
sens = [sensitivity(c) for c in cs]
specs = [1-specificity(c) for c in cs]

plt.scatter(specs, sens)
plt.scatter([1-specificity(15)],[sensitivity(15)])
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

plt.plot(cs, [corr_prevalence(c) for c in cs])
plt.xlabel("Cutoff")
plt.ylabel("Corrected Prevalence")
plt.scatter([15],[raw_prevalance(15)])
plt.show()

