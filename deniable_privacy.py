from scipy.stats import binom
from math import exp, log
from decimal import *
import numpy as np
import scipy
import matplotlib.pyplot as plt


class DeniablePrivacy:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def valid_outputs(self):
        return range(1, self.n)

    def prob_output_appearing(self, a):
        n = self.n
        p = self.p
        assert a > 0 and a < n
        if a > 1 and a < n - 1:
            prob = binom.pmf(k=a, n=n, p=p)
        elif a == 1:
            prob = binom.pmf(k=0, n=n, p=p) + binom.pmf(k=1, n=n, p=p)
        elif a == n - 1:
            prob = binom.pmf(k=n - 1, n=n, p=p) + binom.pmf(k=n, n=n, p=p)
        return prob

    def get_min_epsilon_for_count(self, a):
        n = self.n
        p = self.p
        ## a is the output of the full mechanism
        assert a > 0 and a < n
        if a > 1 and a < n - 1:
            ## t_i = 0
            numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))
            ## t_i = 1
            denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))

        elif a == 1:
            ## t_i = 0  -->  real count could be either 0 or 1
            numerator = Decimal(binom.pmf(k=0, n=n - 1, p=p)) + Decimal(
                binom.pmf(k=1, n=n - 1, p=p)
            )
            ## t_i = 1  -->  count can only be 1, n-1 count can only be 0
            denominator = Decimal(binom.pmf(k=0, n=n - 1, p=p))

        elif a == n - 1:
            ## t_i = 0  -->  count can only be n-1
            numerator = Decimal(binom.pmf(k=n - 1, n=n - 1, p=p))
            ## t_i = 1  -->  count can either be n-2 or n-1
            denominator = Decimal(binom.pmf(k=n - 2, n=n - 1, p=p)) + Decimal(
                binom.pmf(k=n - 1, n=n - 1, p=p)
            )

        else:
            raise ValueError("invalid count")

        return abs(log(numerator) - log(denominator))

    def is_eps_deniable_private(self, a, eps, verbose=False):
        min_eps = self.get_min_epsilon_for_count(a)
        if verbose:
            print("a, min_eps: ", a, min_eps)
        return min_eps <= eps

    def get_success_rate(self, eps, verbose=False):
        expected_success_rate = 0
        for a in range(1, self.n):
            expected_success_rate += self.prob_output_appearing(
                a
            ) * self.is_eps_deniable_private(a, eps)

        return expected_success_rate

    def get_min_eps_slow(self, failure_rate):
        ## returns the minimum epsilon that is compatible with
        ## a given failure rate
        eps = -1
        for a in self.valid_outputs():
            cand_eps = self.get_min_epsilon_for_count(a)
            if 1 - failure_rate < self.get_success_rate(cand_eps):
                eps = max(cand_eps, eps)
        return eps

    # def get_min_eps(self, failure_rate):
    #     ## returns the minimum epsilon that is compatible with
    #     ## a given failure rate
    #     eps = 1e9  ## dummy value. effectively infinity
    #     for a in self.valid_outputs():
    #         cand_eps = self.get_min_epsilon_for_count(a)
    #         if 1 - failure_rate < self.get_success_rate(eps):
    #             eps = min(cand_eps, eps)
    #     return eps


p = 0.5
delta = 1e-9
# eps = np.arange(0.1, 5.0, )
n_vals = []
eps_vals = []
for n in range(3, 60):
    den_priv = DeniablePrivacy(n=n, p=p)
    eps = den_priv.get_min_eps_slow(delta)
    n_vals.append(n)
    eps_vals.append(eps)

print(n_vals)
print(eps_vals)

n_vals = range(3,60)


eps_vals_10_3 = eps_vals

plt.plot(n_vals, eps_vals, marker="o", label="minimum epsilon for delta=10^-9")
# plt.plot(n_vals, eps_vals_10_6, marker="o", label="minimum epsilon for delta=10^-6")
# plt.plot(n_vals, eps_vals_10_3, marker="o", label="minimum epsilon for delta=10^-3")
plt.title("Minimum Achievable Epsilon for Various Delta Values")
plt.xlabel("n")
plt.ylabel("epsilon")
plt.legend()
plt.show()
