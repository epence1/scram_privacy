from scipy.stats import binom
from math import exp, log
from decimal import *
import numpy as np
import matplotlib.pyplot as plt

## Equivalent to "Possible Innocence" idea as described here: https://www.cs.utexas.edu/~shmat/courses/cs395t_fall04/crowds.pdf
## "Probable Innocence" likely prevents attackers from acting on their suspicions according to paper

class KDeniablePrivacy:
    def __init__(self, n, p, k):
        self.n = n
        self.p = p
        self.k = k

    def valid_outputs(self):
        ## Never output a count in {0,...,k-1} and {n-k+1,...,n}
        return range(self.k, self.n - self.k + 1) #TODO CHECK THIS

    def prob_output_appearing(self, a):
        n = self.n
        p = self.p
        k = self.k
        ## Filter out a={0,...,k-1}  and a={n-k+1,...,n} because these outputs are not possible
        assert a > k-1 and a < n-k+1 #TODO CHECK THIS
        if a > k and a < n - k: #TODO CHECK THIS
            prob = binom.pmf(k=a, n=n, p=p)
        elif a == k:
            ## If we see output of k, the real output is either some value {0,...,k}
            prob = 0
            for i in range(0,k+1): #TODO CHECK THIS
                prob += binom.pmf(k=i, n=n, p=p)
        elif a == n - k:
            ## If we see output of n-k, the real output is some value {n-k,...,n}
            prob = 0
            for i in range(n-k,n+1): #TODO CHECK THIS
                prob += binom.pmf(k=i, n=n, p=p)
        return prob

    def get_min_epsilon_for_count(self, a):
        n = self.n
        p = self.p
        k = self.k
        ## a is the output of the full mechanism
        assert a > k-1 and a < n-k+1 #TODO CHECK THIS
        if a > k and a < n - k: #TODO CHECK THIS
            ## t_i = 0 --> total count is exactly count=a
            ## The remaining n-1 inputs must produce exactly count=a
            numerator = Decimal(binom.pmf(k=a, n=n-1, p=p))
            
            ## t_i = 1 --> total count is exactly count=a
            ## The remaining n-1 inputs must be produce exactly count=a-1
            denominator = Decimal(binom.pmf(k=a - 1, n=n-1, p=p))

        elif a == k:
            ## t_i = 0  -->  total count could be any value {0,...,k}
            ## The remaining n-1 inputs produce any count={0,...,k}
            numerator = 0
            for i in range(0,k+1): #TODO CHECK THIS
                numerator += Decimal(binom.pmf(k=i, n=n-1, p=p))
            
            ## t_i = 1  -->  total count can be any value {1, k}
            ## The remaining n-1 inputs must produce count={0,...,k-1}
            denominator = 0 
            for i in range(0,k): #TODO CHECK THIS
                denominator += Decimal(binom.pmf(k=i, n=n-1, p=p))

        elif a == n - k:
            ## t_i = 0  -->  total count can be any value {n-k,...,n-1}
            ## The remaining n-1 inputs must produce a={n-k,...,n-1}
            numerator = 0
            for i in range(n-k,n): #TODO CHECK THIS
                numerator += Decimal(binom.pmf(k=i, n=n-1, p=p))
            
            ## t_i = 1  -->  total count can be any value {n-k,...,n}
            ## The remaining n-1 inputs produce {n-k-1,...,n-1}
            denominator = 0
            for i in range(n-k-1,n): #TODO CHECK THIS
                denominator += Decimal(binom.pmf(k=i, n=n-1, p=p))

        else:
            raise ValueError("invalid count")
        ## Compute value of epsilon required to bound ratios of num/denom and denom/num both
        return abs(log(numerator) - log(denominator))

    def is_eps_deniable_private(self, a, eps, verbose=False):
        '''
        Checks that the minimum epsilon required to make count=a deniably private
        is less than the epsilon specified by k-Deniable Privacy.
        '''
        min_eps = self.get_min_epsilon_for_count(a)
        if verbose:
            print("a, min_eps: ", a, min_eps)
        return min_eps <= eps

    def get_success_rate(self, eps, verbose=False):
        '''
        Computes a weighted likelihood of being deniably private
        across all possible counts, weighting each count by its 
        probability of occurance.
        '''
        expected_success_rate = 0
        priv_range = [-1,-1]
        for a in self.valid_outputs(): # TODO check this
            is_private = self.is_eps_deniable_private(a, eps, verbose)
            if is_private and priv_range[0]==-1:
                priv_range[0]=a
            if not is_private and priv_range[0]!=-1 and priv_range[1]==-1:
                priv_range[1]=a-1
            
            expected_success_rate += self.prob_output_appearing(a) * is_private
            
        if priv_range[1] == -1:
            priv_range[1] = self.n-1
        return expected_success_rate, priv_range

    def get_min_eps_slow(self, failure_rate):
        '''
        Computes smallest epsilon that meets the delta and returns this minimum epsilon.
        Does not necessarily support the full range of outputs 1->n-1
        '''
        eps = 1e9 
        output_range = [-1,-1]
        for a in self.valid_outputs():
            ## Compute epsilon value necessary to bound this specific a
            cand_eps = self.get_min_epsilon_for_count(a)
            ## Check the performance of this epsilon across all counts: Does it bound them?
            success_rate, private_range = self.get_success_rate(cand_eps)
            ## If success rate required is less than success rate achieved, cand_eps suffices
            if 1 - failure_rate < success_rate:
                ## If we are updating epsilon, also update private range
                if cand_eps < eps:
                    output_range=private_range
                    eps = cand_eps
    
        return eps, output_range

    def get_eps_full_range(self,):
        eps = 0
        for a in self.valid_outputs():
            cand_eps = self.get_min_epsilon_for_count(a)
            ## Take max over all of the mins to determine what epsilon is required to protect even the most unlikely of outcomes
            eps = max(cand_eps, eps)
        return eps

#######################################################################
K=5
NMIN = K*2+1 # TODO Is this necessary?
NMAX = 100
p = 0.5
delta = 1e-9
n_vals = range(NMIN,NMAX)
k_den_eps_vals = []
big_eps_vals = []

## Iterate and compute epsilons
for n in n_vals:
    print("testing n: ", n)
    k_den_priv = KDeniablePrivacy(n=n, p=p, k=K)
    eps, output_range = k_den_priv.get_min_eps_slow(delta)
    k_den_eps_vals.append(eps)

    max_eps = k_den_priv.get_eps_full_range()
    big_eps_vals.append(max_eps)

print(n_vals)
print(k_den_eps_vals)

## Plot Eps for Deltas
plt.plot(n_vals, k_den_eps_vals, marker="o", label="K-Deniable: minimum epsilon for delta=10^-9")
plt.plot(n_vals, big_eps_vals, marker="x", label="K-Deniable: minimum epsilon for delta=0")
plt.title("Min Epsilon for Various N")
plt.xlabel("n")
plt.ylabel("epsilon")
plt.ylim(-1,10)
plt.legend()
plt.show()

# ## Plot Probability mass of non private cases
# plt.plot(n_vals, ends, marker="x", label="Deniable: Combined Probability Mass of c=1, c=n-1")
# plt.axhline(y = 1e-9, color = 'r', linestyle = '-', label="Delta=1e-9")
# plt.title("Combined Probability Mass of c=1, c=n-1 for Various N")
# plt.xlabel("n")
# plt.ylabel("log(probability)")
# plt.yscale("log")
# plt.legend()
# plt.show()

# ## Plot Probability Density for Deniable Privacy
# N = 20
# den_priv = DeniablePrivacy(n=N, p=0.5)
# probability_density_x = []
# probability_density_y = []

# for a in range(1,N):
#     probability_density_x.append(a)
#     probability_density_y.append(den_priv.prob_output_appearing(a))

# text = 'Probability of count=1 is '+str(probability_density_y[0])+'.\n Probability of count=(n-1) is '+str(probability_density_y[0])
# plt.plot(probability_density_x, probability_density_y, marker="o", label="Probability Density")
# plt.xticks(np.arange(min(probability_density_x), max(probability_density_x)+1, 1.0))

# # plt.text(probability_density_x[0]+0.2, probability_density_y[0]*1.02, text, ha='center')
# plt.title("Deniable Privacy: Probability Density Function of Possible Outputs for N="+str(N))
# plt.xlabel("count=a")
# plt.ylabel("Probability of seeing output count=a")
# plt.legend()
# plt.show()