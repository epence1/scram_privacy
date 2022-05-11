from scipy.stats import binom
from math import exp, log
from decimal import *
import numpy as np
import matplotlib.pyplot as plt

class ENPPrivacy:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.count_range = [0,self.n]

    def valid_outputs(self):
        # No restrictions on what can be output
        return range(0, self.n + 1)
        
    def prob_output_appearing(self, a):
        """
        Computes probability of seeing count=a, given n and p
        Assumes count is a binomial random variable

        Args:
        a <int> : count of 1s, in range {0,1,...,n}

        Returns:
        prob <float> : probability
        """
        prob = binom.pmf(k=a, n=self.n, p=self.p)
        return prob

    def get_min_epsilon_for_count(self, a):
        '''
        Computes the epsilon value, for a given count, such that
        P[Count=a|t_i=0]/P[Count=a|t_i=1] = e^eps AND
        P[Count=a|t_i=1]/P[Count=a|t_i=0] = e^eps

        Any larger epsilon value will ensure eNP privacy

        Args:
        a <int> : count of 1s in input

        Returns:
        eps_lower_bound <float> : just below the smallest epsilon value to ensure eNP holds
        '''
        if a == 0 or a == self.n:
            # There is no epsilon that can make these two cases private so we make it extremely large
            return 1e10 
        
        numerator = Decimal(binom.pmf(k=a, n=self.n - 1, p=self.p))  # fixes nth input as 0
        denominator = Decimal(binom.pmf(k=a - 1, n=self.n - 1, p=self.p))  # fixes nth input as 1
        
        # The below expression is derived as follows:
        # A/B < e^eps AND B/A < e^eps
        # A/B > e^-eps AND B/A > e^-eps
        # log(A/B) < eps AND log(A/B) > -eps
        # -eps < log(A/B) < eps
        # -eps < log(A) - log(B) < eps
        # |log(A) - log(B)| < eps
        eps_lower_bound = abs(log(numerator) - log(denominator))
        return eps_lower_bound
    
    def is_eps_noiseless_private(self, eps, a, verbose=False):
        """
        Computes indicator variable I_a
        Treats count as Binomial(n'=n-1, p)
        I_a=1 when ratios P[Count=a|t_i=0]/P[Count=a|t_i=1] and P[Count=a|t_i=1]/P[Count=a|t_i=0] are both less than e^eps
        I_a=0 when either ratio is larger than the privacy threshold e^eps

        Args:
        a <int> : count of 1s

        Returns:
        indicator <bool> : indicator random variable that is true only when eNP privacy constraint is met
        """
        # Count of n or zero can never be private; you know everyone's inputs
        if a >= self.n or a == 0:
            return 0

        # Make sure count being checked is positive
        assert a > 0

        min_eps = self.get_min_epsilon_for_count(a)
        if verbose:
            print("a, min_eps: ", a, min_eps)
        
        return min_eps<=eps

    # Epsilon -> Delta
    # This function answers the question: how big does delta need to be given my epsilon, n, p?
    # More simply, which count values will I need to accept as non-private?
    def get_success_rate(self, eps):
        """
        Computes the expected value of an indicator variable I.

        I is 1 when, for all count=a where 0<a<n and an integer, the ratios
        P[Count=a|t_i=0]/P[Count=a|t_i=1] and P[Count=a|t_i=1]/P[Count=a|t_i=0]
        are both less than e^eps.
        The expected value of I is the weighted sum of indicator random variables
        I_a, which equal 1 when a particular count of a is eNP and 0 otherwise.
        Each I_a is weighted by its probability of occurance.

        E[I] represnts the probability eNP holds for any count=a, where 0<a<n and an integer,
        that is sampled from the Binomial(n,p) distribution.

        Args:

        Returns:
        <tuple> of expected_value_of_indicator <float>, private_range <tuple> : private_range is of form (smallest private a, largest private a)
        """
        expected_success_rate = 0
        is_eps_private_flag = False
        flag_switched = False
        lower_a, upper_a = -1, -1

        for a in range(self.count_range[0], self.count_range[1]+1):  # +1 because range func is non-inclusive
            if not flag_switched:
                # If count=a can be eNP and flag is false
                if self.is_eps_noiseless_private(eps, a) and not is_eps_private_flag:
                    # Establish lower bound on acceptable counts
                    lower_a = a
                    # Set flag to true, indicating we expect a continuous region of acceptable a values (counts)
                    is_eps_private_flag = True

                # If count=a is not eNP and previous counts were eNP
                elif not self.is_eps_noiseless_private(eps, a) and is_eps_private_flag:
                    # Establish previous a value as largest eNP count
                    upper_a = a - 1
                    # Set flag back to false
                    is_eps_private_flag = False
                    # Indicate the continuous region of acceptable counts has terminated
                    flag_switched = True
            else:
                # Should be not private since continuous region terminated
                if self.is_eps_noiseless_private(eps, a):
                    print("going back and forth...")
                    print("good a range: [", lower_a, ",", upper_a, "]")
                    print("new a: ", a)
                    assert 1 == 0

            # Expected value of indicator is sum of P[Count=a] * I[Count=a]
            expected_success_rate += self.prob_output_appearing(a) * self.is_eps_noiseless_private(eps, a)

        ## I think this logic makes sense but im not sure we need it if we check all the way to count of n
        ## since n is never private
        if upper_a == -1 and lower_a != -1:
            # If the last count we checked was eNP private, make sure we set the upper bound on a to be n-1
            upper_a = self.n - 1

        private_range = [lower_a, upper_a]
        return expected_success_rate, private_range

    def get_min_eps_slow(self, failure_rate):
        '''
        Computes the smallest value of epsilon that can bound every count a={1,...,n-1}

        Does so by taking the maximum over all minimum epsilons for each individual count a

        Args:
        None

        Returns:
        max_eps <float>
        '''

        # Now we take a max over all the min epsilons
        # This ensures we find the smallest eps that can bound the ratio with the given delta
        # DO WE NEED TO NORMALIZE STUFF????
        eps = 1e9 
        output_range = [-1,-1]
        for a in range(0, self.n + 1):
            cand_eps = self.get_min_epsilon_for_count(a)
            success_rate, private_range = self.get_success_rate(cand_eps)
            if 1 - failure_rate < success_rate:
                ## If we are updating epsilon, also update range
                if cand_eps < eps:
                    output_range=private_range
                eps = min(cand_eps, eps)
        return eps, output_range


    # Delta -> Epsilon
    def compute_epsilon_given_delta_full_range(
        self,
        delta, 
        eps_search_space=[0.01, 10.0, 0.01], 
        ):
        '''
        Computes smallest epsilon for a given delta (eNP failure rate) 
        while applying the constraint that all counts 1,...,n-1 must be 
        eNP private for that epsilon.
        
        It does this by identifying the epsilon value that minimizes the 
        difference between the specfied delta and the actual delta.

        The significance here is that a practioner might ask, "how do I set epsilon
        to ensure eNP holds x% of the time?"

        Args:
        delta <float> : acceptable failure rate of eNP
        eps_search_space <list> : list of 2 floats specifying the [lower, upper] bounds on potential epsilon values

        Returns:
        eps_out <float> : epsilon required to achieve specfied delta
        '''
        for eps in np.arange(eps_search_space[0], eps_search_space[1], eps_search_space[2]):
            #print("checking eps", eps)
            exp_success_rate, private_range = self.get_success_rate(eps=eps)
            #print("range, exp success rate", private_range, exp_success_rate)
            if exp_success_rate > 1:
                # expected value is greater than 1 sometimes; e.g. 1.0000000005
                exp_success_rate = 1
            if private_range == (1, self.n - 1):
                computed_delta = 1 - exp_success_rate
                ## LESS THAN OR EQUAL TO??
                if computed_delta <= delta:
                    print("computed_delta, eps : ", computed_delta, eps)
                    return eps
        raise ValueError(
            "Not satisfiable: There is no epsilon in the given range that can produce the given delta while ensuring 1->n-1 is eNP private"
            )


###
n_vals = range(29,32)  # Observation: smallest n with reasonable delta is 30 or 31
p_vals = np.arange(0.1, 1.0, 0.1) 
# According to NIST (https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-future-work-open-challenges)
# Epsilon values between 0 and 5 are strong
eps_vals = np.arange(0.1, 5.1, 0.1)
###

## Uncomment if you want the probability mass across counts for a given n
# for n in range(3,7):
#     for a in range(0, n + 1):
#         print("n, a, prob count appears: ",n, a, binom.pmf(k=a, n=n, p=0.1))

## Uncomment to see min epsilon that ensures 1 -> n-1 is eNP
# for n in range(3, 10):
#     print("checking n =", n)
#     enp_priv = ENPPrivacy(n=n, p=0.1)
#     print("Minimum Bounding Epsilon for all counts: ", enp_priv.compute_min_epsilon())

## Uncomment if you want to verify compute_epsilon_given_delta_full_range() works
# enp_priv = ENPPrivacy(n=50, p=0.5)
# print(enp_priv.compute_epsilon_given_delta_full_range(delta=1e-9))
# # # By setting epsilon at 3.8, just below the required threshold, we see the full range is not achieved
# print(enp_priv.get_success_rate(eps=3.8))

## Uncomment if you want to see the private ranges given n, p, eps
# for n in n_vals:
#     for p in p_vals:
#         enp_priv = ENPPrivacy(n=n, p=p)
#         for eps in eps_vals:
#             # Delta represents the probability that eNP does not hold
#             # Delta = 1-E[I]
#             # According to (https://www.ftc.gov/system/files/documents/public_comments/2017/11/00023-141742.pdf)
#             # Delta should be 1/(1 billion) = 1e-9
#             exp_val, private_range = enp_priv.get_success_rate(eps)
#             delta = 1 - exp_val
#             if delta <= 1e-9:
#                 print("good a range: [", private_range[0], ",", private_range[1], "]")
#                 print("(n,p,eps,delta) =",(n, p, eps, delta), "\n")

# ## Plot Probability Density for eNP Privacy
# N = 20
# P=0.5
# den_priv = ENPPrivacy(n=N, p=P)
# probability_density_x = []
# probability_density_y = []

# for a in range(0,N+1):
#     probability_density_x.append(a)
#     probability_density_y.append(den_priv.prob_output_appearing(a))

# text = 'Probability of count=0 is '+str(probability_density_y[0])+'.\n Probability of count='+str(N)+' is '+str(probability_density_y[0])
# plt.plot(probability_density_x, probability_density_y, marker="o", label="Probability Density")
# plt.xticks(np.arange(min(probability_density_x), max(probability_density_x)+1, 1.0))

# plt.text(probability_density_x[0]+3, probability_density_y[0]+0.2, text, ha='center')
# plt.title("eNP: Probability Density Function of Possible Outputs for N="+str(N))
# plt.xlabel("Count=a")
# plt.ylabel("Probability of seeing output count=a")
# plt.legend()
# plt.show()

# ## Plot size if eposilon for static N, p
# N = 20
# P=0.5
# enp_priv = ENPPrivacy(n=N, p=P)
# count_vals = []
# eps_vals = []

# for a in range(0,N+1):
#     min_eps_for_count = enp_priv.get_min_epsilon_for_count(a)
#     count_vals.append(a)
#     eps_vals.append(min_eps_for_count)

# plt.title("eNP: Minimum Epsilon Required to Protect Each Possible Output, given N="+str(N)+" P="+str(P))
# plt.plot(count_vals, eps_vals, marker="o", label="eNP: Minimum Epsilon")
# plt.xlabel("Count=a")
# plt.ylabel("Min epsilon to protect count=a")
# plt.xticks(np.arange(min(count_vals), max(count_vals)+1, 1.0))
# plt.ylim(-1,5)
# plt.legend()
# plt.show()
