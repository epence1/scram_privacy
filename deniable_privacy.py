from cv2 import NORM_MINMAX
from scipy.stats import binom
from math import exp, log
from decimal import *
import numpy as np
import scipy
import matplotlib.pyplot as plt

## Equivilent to "Possible Innocence" idea as described here: https://www.cs.utexas.edu/~shmat/courses/cs395t_fall04/crowds.pdf
## "Probable Innocence" likely prevents attackers from acting on their suspicions according to paper
## Does this mean "Possible Innocence" will draw action from attackers?

class DeniablePrivacy:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def valid_outputs(self):
        ## Never output count of zero or count of n
        ## Both cases reveal the inputs of all participants
        return range(1, self.n)

    def prob_output_appearing(self, a):
        n = self.n
        p = self.p
        ## Filter out a==0 and a==n because these outputs are not possible
        assert a > 0 and a < n
        if a > 1 and a < n - 1:
            prob = binom.pmf(k=a, n=n, p=p)
        elif a == 1:
            ## If we see output of 1, the real output is either 1 or 0
            prob = binom.pmf(k=0, n=n, p=p) + binom.pmf(k=1, n=n, p=p)
        elif a == n - 1:
            ## If we see output of n-1, the real output is either n-1 or n
            prob = binom.pmf(k=n - 1, n=n, p=p) + binom.pmf(k=n, n=n, p=p)
        return prob

    def get_min_epsilon_for_count(self, a):
        n = self.n
        p = self.p
        ## a is the output of the full mechanism
        assert a > 0 and a < n
        if a > 1 and a < n - 1:
            ## t_i = 0
            ## The remaining n-1 inputs must produce exactly count=a
            numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))
            ## t_i = 1: 
            ## The remaining n-1 inputs must be produce eaxctly count=a-1
            denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))

        elif a == 1:
            ## t_i = 0  -->  real count could be either 0 or 1
            ## The remaining n-1 inputs produce either count=0 or count=1
            numerator = Decimal(binom.pmf(k=0, n=n - 1, p=p)) + Decimal(
                binom.pmf(k=1, n=n - 1, p=p)
            )
            ## t_i = 1  -->  count can only be 1, n-1 count can only be 0
            ## The remaining n-1 inputs must produce count=0
            denominator = Decimal(binom.pmf(k=0, n=n - 1, p=p))

        elif a == n - 1:
            ## t_i = 0  -->  count can only be n-1
            ## The remaining n-1 inputs must produce a=n-1
            ## If we are given t_i=0, we know the full count can never equal n
            numerator = Decimal(binom.pmf(k=n - 1, n=n - 1, p=p))
            ## t_i = 1  -->  count can either be n-2 or n-1
            ## The remaining n-1 inputs produce either a=n-1 or a=n-2
            ## Since count=n-1 is output for both count=n and count=n-1
            ## If we are given t_i=1, we know either outcome is achievable
            denominator = Decimal(binom.pmf(k=n - 2, n=n - 1, p=p)) + Decimal(
                binom.pmf(k=n - 1, n=n - 1, p=p)
            )

        else:
            raise ValueError("invalid count")
        ## Compute value of epsilon required to bound ratios of num/denom and denom/num both
        return abs(log(numerator) - log(denominator))

    def is_eps_deniable_private(self, a, eps, verbose=False):
        '''
        Checks that the minimum epsilon required to make count=a eNP 
        is less than the epsilon specified by eNP
        '''
        min_eps = self.get_min_epsilon_for_count(a)
        if verbose:
            print("a, min_eps: ", a, min_eps)
        ## Only if min_eps is less than eps because technically min_eps sits on the threshold 
        ## between private and nnot private under the specified eps
        return min_eps < eps

    def get_success_rate(self, eps, verbose=False):
        '''
        Computes a weighted likelihood of being deniably private
        across all possible counts, weighting each count by its 
        probability of occurance
        '''
        expected_success_rate = 0
        for a in range(1, self.n):
            expected_success_rate += self.prob_output_appearing(
                a
            ) * self.is_eps_deniable_private(a, eps, verbose)

        return expected_success_rate

    def get_min_eps_slow(self, failure_rate):
        ## returns the minimum epsilon that is compatible with
        ## a given failure rate
        ## does not necessarily support the full range of outputs 1->n-1
        eps = 1e9 
        for a in self.valid_outputs():
            ## Compute epsilon value necessaryto bound this specific a
            cand_eps = self.get_min_epsilon_for_count(a)
            ## If success rate required is less than success rate achieved cand_eps suffices
            if 1 - failure_rate < self.get_success_rate(cand_eps):
                ## Take minimum so that we know smallest epsilon that provides any privacy guarantees????
                eps = min(cand_eps, eps)
        return eps

    def get_eps_full_range(self, failure_rate):
        eps = 0
        for a in self.valid_outputs():
            cand_eps = self.get_min_epsilon_for_count(a)
            ## Take max over all of the mins
            eps = max(cand_eps, eps)
        ## If this max meets the failure rate constraint, then the whole range is supported
        if 1 - failure_rate < self.get_success_rate(eps):
            return eps
        else:
            print("cannot reach whole range with this failure rate")
            #raise ValueError("cannot reach whole range with this failure rate")
            return -1

NMIN = 3
NMAX = 1000
p = 0.5
delta = 1e-9
# eps = np.arange(0.1, 5.0, )
n_vals = []
den_eps_vals = []
big_eps_vals = []
for n in range(NMIN, NMAX):
    print('n')
    den_priv = DeniablePrivacy(n=n, p=p)
    eps = den_priv.get_min_eps_slow(delta)
    max_eps = den_priv.get_eps_full_range(delta)
    n_vals.append(n)
    den_eps_vals.append(eps)
    big_eps_vals.append(max_eps)

print(n_vals)
print(den_eps_vals)
print(big_eps_vals)

n_vals = range(NMIN,NMAX)


plt.plot(n_vals, den_eps_vals, marker="o", label="Deniable: minimum epsilon for delta=10^-9")
plt.plot(n_vals, big_eps_vals, marker="x", label="Deniable: minimum epsilon the full range")
plt.title("Minimum Achievable Epsilon for Various Delta Values")
plt.xlabel("n")
plt.ylabel("epsilon")
plt.ylim([-1, 20])
plt.legend()
plt.show()
