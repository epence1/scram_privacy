import numpy as np
import math
from deniable_privacy import DeniablePrivacy
from eNP_direct import ENPPrivacy
import matplotlib.pyplot as plt


class GaussianNoise:
    """
    For epsilon delta diff privacy
    """

    def __init__(self, epsilon, delta, f_sensitivity=1, mean=0):
        self.f_sens = f_sensitivity
        self.eps = epsilon
        self.delta = delta
        self.mean = mean
        # https://proceedings.mlr.press/v80/balle18a/balle18a.pdf Justifies this sd value
        self.sd = (
            (self.f_sens) * (math.sqrt(2 * np.log(1.25 / self.delta))) / (self.eps)
        )
        # Taken from Dwork 2014 https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        # self.sd = math.sqrt( self.f_sens*np.log(1/self.delta)/self.eps )
        # This paper argues Dwork value is wrong: https://arxiv.org/pdf/1911.12060.pdf

    def get_expected_abs_val(self):
        sumy = 0
        iters = int(1e6)
        for i in range(iters):
            s = self.get_sample()
            sumy += abs(s)
        return sumy / iters

    def get_sample(self):
        return np.random.normal(self.mean, self.sd)


class LaplaceNoise:
    """
    For epsilon diff privacy: Lap(f_sensitivity/eps)
    """

    def __init__(self, epsilon, f_sensitivity=1, mean=0):
        self.eps = epsilon
        self.f_sens = f_sensitivity
        self.mean = mean

        self.b = self.f_sens / self.eps

    def get_expected_abs_val(self):
        return 1 / self.eps

    def get_sample(self):
        return np.random.laplace(loc=self.mean, scale=self.b)

    def get_non_triv_prob(self):
        return math.exp(-0.5 * self.eps)


## Test Noise classes
# EPS = 3
# DELTA = 1e-9
# g = GaussianNoise(epsilon=EPS, delta=DELTA)
# print(g.get_expected_abs_val())
# l = LaplaceNoise(epsilon=EPS)
# print(l.get_expected_abs_val())


NMIN = 3
NMAX = 50
p = 0.5
delta = 1e-9
n_vals = range(NMIN, NMAX)
den_priv_noise_vals = []
diff_priv_noise_vals = []

## Noise Comparison: Diff Privacy vs Deniable Privacy
for n in n_vals:
    print("testing n: ", n)
    # Establish enp privacy so that we can compute exclusive probability of getting c=0, c=n
    enp_priv = ENPPrivacy(n=n, p=p)
    # Count Sensitivity is 1 so its just the mass of these cases
    mass_0_n = enp_priv.prob_output_appearing(0) + enp_priv.prob_output_appearing(n)
    den_priv_noise_vals.append(mass_0_n)

    # Establish Den Privacy so that we can get bounding epsilon
    den_priv = DeniablePrivacy(n=n, p=p)
    den_eps, output_range = den_priv.get_min_eps_slow(delta)

    # Use Den Privacy epsilon for Differential privacy noise function
    l = LaplaceNoise(epsilon=den_eps)
    print(den_eps, l.get_non_triv_prob())
    diff_priv_noise = l.get_expected_abs_val()
    diff_priv_noise_vals.append(diff_priv_noise)


plt.plot(n_vals, den_priv_noise_vals, marker="o", label="Deniable: Expected Noise")
plt.plot(n_vals, diff_priv_noise_vals, marker="x", label="Differential: Expected Noise")
plt.title("Expected Value of Noise: Differential vs Deniable")
plt.xlabel("n")
plt.ylabel("log(expected noise)")
plt.yscale("log")
plt.legend()
plt.show()
