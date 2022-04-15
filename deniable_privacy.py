from tkinter import E
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
        eps = 1e9
        for a in self.valid_outputs():
            cand_eps = self.get_min_epsilon_for_count(a)
            if 1 - failure_rate < self.get_success_rate(eps):
                eps = min(cand_eps, eps)
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
delta = 1e-3
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

n_vals = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
]

eps_vals_10_9 = [
    1.0986122886681096,
    0.0,
    0.4054651081081644,
    0.6931471805599456,
    0.9162907318741549,
    1.0986122886681098,
    1.252762968495368,
    1.3862943611198908,
    1.5040773967762728,
    1.6094379124341014,
    1.7047480922384253,
    1.7917594692280536,
    1.8718021769015927,
    1.9459101490553188,
    2.014903020542267,
    2.079441541679838,
    2.1400661634962717,
    2.1972245773362147,
    2.251291798606495,
    2.3025850929940432,
    2.3513752571634754,
    2.3978952727983724,
    2.4423470353692025,
    2.484906649787993,
    2.525728644308254,
    2.5649493574615345,
    3.3672958299864817,
    2.639057329615266,
    2.674148649426513,
    3.4657359027997288,
    3.4965075614664727,
    2.772588722239771,
    3.55534806148939,
    2.833213344056194,
    2.427748235948034,
    2.456735772821304,
    2.484906649788016,
    2.5123056239760864,
    2.2246235515242994,
    2.251291798606516,
    2.5902671654458373,
    2.302585092994054,
    2.079441541679845,
    2.1041341542701986,
    2.1282317058492595,
    2.1517622032594765,
    1.9694406464655252,
    1.9924301646901768,
    2.219203484054958,
    2.036881927260996,
    1.8827312474337283,
    1.9042374526547725,
    1.9252908618525417,
    1.9459101490552975,
    1.8123787564307747,
    1.832581463748312,
    1.852384091044506,
]

eps_vals_10_6 = [
    1.0986122886681096,
    0.0,
    0.4054651081081644,
    0.6931471805599456,
    0.9162907318741549,
    1.0986122886681098,
    1.252762968495368,
    1.3862943611198908,
    1.5040773967762728,
    1.6094379124341014,
    1.7047480922384253,
    1.7917594692280536,
    1.8718021769015927,
    1.9459101490553188,
    2.014903020542267,
    2.079441541679838,
    2.1400661634962717,
    2.1972245773362147,
    2.251291798606495,
    2.3025850929940432,
    2.3513752571634754,
    2.3978952727983724,
    2.4423470353692025,
    2.0368819272610423,
    2.0794415416798326,
    2.1202635362000937,
    2.1594842493533744,
    1.8718021769015873,
    1.9095425048844294,
    1.9459101490553206,
    1.9810014688665678,
    1.757857917552359,
    1.7917594692280527,
    1.8245492920510475,
    1.8562979903656167,
    1.6739764335716654,
    1.7047480922384448,
    1.7346010553880848,
    1.580450375560833,
    1.6094379124341032,
    1.6376087894008187,
    1.5040773967762817,
    1.5314763709643522,
    1.558144618046569,
    1.5841201044497915,
    1.4663370687934218,
    1.4916548767777336,
    1.516347489368087,
    1.4109869737102478,
    1.4350845252893123,
    1.4586150226995152,
    1.363304842895218,
    1.3862943611198695,
    1.4087672169719951,
    1.3217558399823552,
    1.3437347467010454,
    1.3652409519220896,
]

eps_vals_10_3 = eps_vals

plt.plot(n_vals, eps_vals_10_9, marker="o", label="minimum epsilon for delta=10^-9")
plt.plot(n_vals, eps_vals_10_6, marker="o", label="minimum epsilon for delta=10^-6")
plt.plot(n_vals, eps_vals_10_3, marker="o", label="minimum epsilon for delta=10^-3")
plt.title("Minimum Achievable Epsilon for Various Delta Values")
plt.xlabel("n")
plt.ylabel("epsilon")
plt.legend()
plt.show()
