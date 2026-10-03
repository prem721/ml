# Week 4: EM algorithm - Two Coin Mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

heads = np.array([5, 9, 8, 4, 7, 10, 3, 6, 8, 5])   # heads in each set of n tosses
n = 10

pA, pB, mix = 0.6, 0.5, 0.5   # initial guesses
likelihood = []

for i in range(10):
    # E-Step: responsibility of coin A for each set
    pa = binom.pmf(heads, n, pA)
    pb = binom.pmf(heads, n, pB)
    r = (mix * pa) / (mix * pa + (1 - mix) * pb)

    # M-Step: re-estimate parameters
    mix = np.mean(r)
    pA = np.sum(r * heads) / np.sum(r * n)
    pB = np.sum((1 - r) * heads) / np.sum((1 - r) * n)

    L = np.sum(np.log(mix * pa + (1 - mix) * pb))
    likelihood.append(L)
    print("Iteration:", i + 1, "pA:", round(pA, 3), "pB:", round(pB, 3), "logL:", round(L, 4))

print("\nFinal pA =", round(pA, 3))
print("Final pB =", round(pB, 3))
print("Mixture  =", round(mix, 3))

plt.plot(range(1, 11), likelihood, "o-")
plt.xlabel("Iteration"); plt.ylabel("Log-likelihood"); plt.title("EM convergence")
plt.savefig("em_two_coin.png", dpi=120)
plt.show()
