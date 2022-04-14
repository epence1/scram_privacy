from scipy.stats import binom
from math import exp
from decimal import *
import numpy as np
import scipy


def noiseless_privacy_analysis(n, p, eps):
    '''
    Computes the expected value of an indicator variable I.
    
    I is 1 when, for all count=a where 0<a<n and an integer,the ratios 
    P[Count=a|t_i=0]/P[Count=a|t_i=1] and P[Count=a|t_i=1]/P[Count=a|t_i=0] 
    are both less than e^eps.  
    The expected value of I is the weighted sum of indicator random variables 
    I_a, which equal 1 when a particular count of a is eNP and 0 otherwise.
    Each I_a is weighted by its probability of occurance.
    
    E[I] represnts the probability eNP holds for some count=a, where 0<a<n and an integer, 
    that is sampled from the Binomial(n,p) distribution 

    Args:
    n <int> : number of inputs
    p <float> : probability of an input taking value 1
    eps <float> : epsilon, where e^eps defines our privacy threshold

    Returns:
    <tuple> of private_range <tuple>, expected_value_of_indicator <float> : private_range is of form (smallest private a, largest private a)
    '''
    def prob_count_appearing(a):
        '''
        Computes probability of seeing count=a, given n and p
        Assumes count is a binomial random variable

        Args:
        a <int> : count of 1s, in range {0,1,...,n}

        Returns:
        prob <float> : probability
        '''
        prob = binom.pmf(k=a, n=n, p=p)
        return prob

    def is_eps_noiseless_private(a):
        '''
        Computes indicator variable I_a
        Treats count as Binomial(n'=n-1, p)
        I_a=1 when ratios P[Count=a|t_i=0]/P[Count=a|t_i=1] and P[Count=a|t_i=1]/P[Count=a|t_i=0] are both less than e^eps
        I_a=0 when either ratio is larger than the privacy threshold e^eps

        Args:
        a <int> : count of 1s

        Returns:
        indicator <bool> : indicator random variable that is true only when eNP privacy constraint is met
        '''
        # Count of n or zero can never be private; you know everyone's inputs
        if a >= n or a == 0:
            return 0

        # Make sure count being checked is positive
        assert a > 0
        
        # Set privacy threshold as e^eps
        thresh = Decimal(exp(eps))

        # If we truly want to fix the nth input, dont we have to multiply by prob that is zero or 1?
        numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p)) # fixes nth input as 0
        denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p)) # fixes nth input as 1
        indicator = numerator < thresh * denominator and denominator < thresh * numerator
        return indicator

    expected_val_of_indicator = 0
    is_eps_private_flag = False
    flag_switched = False
    lower_a, upper_a = -1, -1
    
    for a in range(0, n + 1):
        if not flag_switched:
            # If count=a can be eNP and flag is false
            if is_eps_noiseless_private(a) and not is_eps_private_flag:
                # Establish lower bound on acceptable counts
                lower_a = a
                # Set flag to true, indicating we expect a continuous region of acceptable a values (counts)
                is_eps_private_flag = True
            
            # If count=a is not eNP and previous counts were eNP
            elif not is_eps_noiseless_private(a) and is_eps_private_flag:
                # Establish previous a value as largest eNP count
                upper_a = a-1
                # Set flag back to false
                is_eps_private_flag = False
                # Indicate the continuous region of acceptable counts has terminated
                flag_switched = True
        else:
            # Should be not private since continuous region terminated
            if is_eps_noiseless_private(a):
                print("going back and forth...")
                print("good a range: [", lower_a, ",", upper_a, ")")
                print("new a: ", a)
                assert 1 == 0
        
        # Expected value of indicator is sum of P[Count=a] * I[Count=a]
        expected_val_of_indicator += prob_count_appearing(a) * is_eps_noiseless_private(a)
    
    private_range = (lower_a, upper_a)
    # print("good a range: [", lower_a, ",", upper_a, ")")
    return private_range, expected_val_of_indicator

def compute_epsilon_given_delta(delta, p, n, eps_search_space = [0.0, 7.0]):
    '''
    Computes smallest epsilon for a given delats (failure rate) by identifying 
    the epsilon value that minimizes the difference between the specfied delta and 
    the actual delta (a function of epsilon)

    The significance here is that a practioner might ask, "how do I set epsilon
    to ensure eNP holds x% of the time?"

    Args:
    delta <float> : acceptable failure rate of eNP
    p <float> : probability of t_i=1 (for any t_i in input T)
    n <int> : number of inputs, or participants
    eps_search_space <list> : list of 2 floats specifcying the [lower, upper] bounds on potential epsilon values
    
    Returns:
    eps_out <float> : epsilon required to achieve specfied delta
    '''
    # Observation: small changes in epsilon do not seem to change failure likelihood
    def f(eps, delta, p, n):
        private_range, exp_val = noiseless_privacy_analysis(n, p, eps)
        computed_delta = 1-exp_val
        # if computed delta is much smaller than delta, then this is very negative which is better
        return computed_delta - delta
    
    eps_opt = scipy.optimize.fminbound(f, eps_search_space[0], eps_search_space[1], args=(delta, p, n,))
    return eps_opt

def compute_epsilon_given_delta_full_range(delta, p, n, eps_search_space = [0.01, 10.0, 0.01]):
    for eps in np.arange(eps_search_space[0], eps_search_space[1], eps_search_space[2]):
        private_range, exp_val = noiseless_privacy_analysis(n, p, eps)
        if exp_val > 1:
            # expected value is greater than 1 sometimes
            # print("expected value greater than 1, setting to 1")
            # print(exp_val)
            exp_val = 1
        if private_range == (1,n-1):
            computed_delta = 1-exp_val
            if computed_delta <= delta:
                print(computed_delta, eps)
                return eps
    raise ValueError("not satisfiable")


n_vals = range(80,81) # smallest n with reasonable delts is 30 or 31
p_vals = np.arange(0.1, 1.0, 0.1)
# According to NIST (https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-future-work-open-challenges)
# Epsilon values between 0 and 5 are strong
eps_vals = np.arange(0.1, 5.1, 0.1)

compute_epsilon_given_delta_full_range(delta = 1e-9, p=0.5, n=50)
print(compute_epsilon_given_delta(delta = 1e-9, p=0.5, n=50))
print(noiseless_privacy_analysis(n=50,p=0.5,eps=3.8))


# for n in n_vals:
#     for p in p_vals:
#         for eps in eps_vals:
#             # Delta represents the probability that eNP does not hold
#             # Delta = 1-E[I]
#             # According to (https://www.ftc.gov/system/files/documents/public_comments/2017/11/00023-141742.pdf)
#             # Delta should be 1/(1 billion) = 1e-9
#             private_range, exp_val = noiseless_privacy_analysis(n, p, eps)
#             delta = 1 - exp_val
#             if delta <= 1e-9:
#                 print("good a range: [", private_range[0], ",", private_range[1], ")")
#                 print("(n,p,eps,delta) =",(n, p, eps, delta), "\n")

