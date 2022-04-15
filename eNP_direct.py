from scipy.stats import binom
from math import exp, log
from decimal import *
import numpy as np
import scipy


def noiseless_privacy_analysis(n, p, eps, exclude_all_zeros=False, exclude_all_ones=False):
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
    n <int> : number of inputs
    p <float> : probability of an input taking value 1
    eps <float> : epsilon, where e^eps defines our privacy threshold

    Returns:
    <tuple> of private_range <tuple>, expected_value_of_indicator <float> : private_range is of form (smallest private a, largest private a)
    """
    
    # Include/exclude edge cases based on inputs booleans
    # (excluding non-private endpoints by default)
    count_range = [0,n]
    if exclude_all_zeros:
        count_range[0]=1
    if exclude_all_ones:
        count_range[1]=n-1

    def prob_count_appearing(a):
        """
        Computes probability of seeing count=a, given n and p
        Assumes count is a binomial random variable

        Args:
        a <int> : count of 1s, in range {0,1,...,n}

        Returns:
        prob <float> : probability
        """
        prob = binom.pmf(k=a, n=n, p=p)
        return prob

    def is_eps_noiseless_private(a):
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
        if a >= n or a == 0:
            return 0

        # Make sure count being checked is positive
        assert a > 0

        # Set privacy threshold as e^eps
        thresh = Decimal(exp(eps))

        # Since its Binomial, fixinng one input has no impact on probability of others
        numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))  # fixes nth input as 0
        denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))  # fixes nth input as 1
        indicator = (
            numerator < thresh * denominator and denominator < thresh * numerator
        )
        return indicator

    expected_success_rate = 0
    is_eps_private_flag = False
    flag_switched = False
    lower_a, upper_a = -1, -1

    for a in range(count_range[0], count_range[1]+1):  # +1 because range func is non-inclusive
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
                upper_a = a - 1
                # Set flag back to false
                is_eps_private_flag = False
                # Indicate the continuous region of acceptable counts has terminated
                flag_switched = True
        else:
            # Should be not private since continuous region terminated
            if is_eps_noiseless_private(a):
                print("going back and forth...")
                print("good a range: [", lower_a, ",", upper_a, "]")
                print("new a: ", a)
                assert 1 == 0

        # Expected value of indicator is sum of P[Count=a] * I[Count=a]
        expected_success_rate += prob_count_appearing(a) * is_eps_noiseless_private(a)

    if upper_a == -1 and lower_a != -1:
        # If the last count we checked was eNP private, make sure we set the upper bound on a to be n-1
        upper_a = n - 1
    
    # If we exclude these two edge cases from the analysis, we must normalize the probabilities
    # so that they sum to one
    # The reason we might want to exclude these cases is that they will never be private, resulting
    # in an "inflated" eNP failure rate.
    # We are most interested in knowing, specifically in the non-edge cases, what is the likelihood eNP holds.
    # If we fail to acknowledge that these two edge cases often account for a substantial component of
    # the eNP failure rate, then we discount its guarantees.
    if exclude_all_zeros and exclude_all_ones:
        expected_success_rate /= 1 - binom.pmf(k=0, n=n, p=p) - binom.pmf(k=n, n=n, p=p)
    elif exclude_all_zeros and not exclude_all_ones:
        expected_success_rate /= 1 - binom.pmf(k=0, n=n, p=p)
    elif exclude_all_ones and not exclude_all_zeros:
        expected_success_rate /= 1 - binom.pmf(k=n, n=n, p=p)

    private_range = (lower_a, upper_a)
    # print("good a range: [", lower_a, ",", upper_a, "]")
    return private_range, expected_success_rate

def compute_epsilon_given_delta_full_range(
    delta, 
    p, 
    n, 
    eps_search_space=[0.01, 10.0, 0.01], 
    exclude_all_zeros=False, 
    exclude_all_ones=False,
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
    p <float> : probability of t_i=1 (for any t_i in input T)
    n <int> : number of inputs, or participants
    eps_search_space <list> : list of 2 floats specifying the [lower, upper] bounds on potential epsilon values

    Returns:
    eps_out <float> : epsilon required to achieve specfied delta
    '''
    for eps in np.arange(eps_search_space[0], eps_search_space[1], eps_search_space[2]):
        #print("checking eps", eps)
        private_range, exp_success_rate = noiseless_privacy_analysis(
            n=n, 
            p=p, 
            eps=eps, 
            exclude_all_zeros=exclude_all_zeros,
            exclude_all_ones=exclude_all_ones,
            )
        #print("range, exp success rate", private_range, exp_success_rate)
        if exp_success_rate > 1:
            # expected value is greater than 1 sometimes; e.g. 1.0000000005
            exp_success_rate = 1
        if private_range == (1, n - 1):
            computed_delta = 1 - exp_success_rate
            if computed_delta <= delta:
                #print("computed_delta, eps : ", computed_delta, eps)
                return eps
    raise ValueError(
        "Not satisfiable: There is no epsilon in the given range that can produce the given delta while ensuring 1->n-1 is eNP private"
        )


def compute_min_epsilon(p, n):
    '''
    Computes the smallest value of epsilon that can bound every count a={1,...,n-1}

    Does so my taking the maximum over all minimum epsilons for each individual count a

    Args:
    p <float> : probability some input t_i equals 1
    n <int> : number of inputs, or participants

    Returns:
    max_eps <float>
    '''
    def get_min_epsilon(a):
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
        if a == 0 or a == n:
            # There is no epsilon that can make these two cases private
            return -1e10 # Making nagative so that it never satisfies the max
        
        numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))  # fixes nth input as 0
        denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))  # fixes nth input as 1
        
        # The below expresion is derived as follows:
        # A/B < e^eps AND B/A < e^eps
        # A/B < e^eps AND A/B > e^-eps
        # log(A/B) < eps AND log(A/B) > -eps
        # -eps < log(A/B) < eps
        # -eps < log(A) - log(B) < eps
        # |log(A) - log(B)| < eps
        eps_lower_bound = abs(log(numerator) - log(denominator))
        return eps_lower_bound

    # Now we take a max over all the min epsilons
    # This ensures we find the smallest eps that can bound
    # every count in the range
    max_eps = 0
    for a in range(0, n + 1):
        print("count, min epsilon : ", a, get_min_epsilon(a))
        max_eps = max(max_eps, get_min_epsilon(a))
    return max_eps

###
n_vals = range(2,80)  # Observation: smallest n with reasonable delta is 30 or 31
p_vals = np.arange(0.1, 1.0, 0.1) 
# According to NIST (https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-future-work-open-challenges)
# Epsilon values between 0 and 5 are strong
eps_vals = np.arange(0.1, 5.1, 0.1)
###

## Uncomment to test compute_min_epsilon()
for n in range(3, 10):
    print("checking n =", n)
    print("Minimum Bounding Epsilon for all counts: ", compute_min_epsilon(p=0.1, n=n))

## Uncomment if you want to verify compute_epsilon_given_delta_full_range() works
# compute_epsilon_given_delta_full_range(delta=1e-9, p=0.5, n=50)
# print(noiseless_privacy_analysis(n=50, p=0.5, eps=3.9))

## Uncomment if you want to see how delta changes drastically when you exclude the two edge cases
# for n in range(80, 83):
#     print("checking n", n)
#     try:
#         eps_with_zeros_and_ones = compute_epsilon_given_delta_full_range(
#             delta=1e-9, 
#             p=0.5, 
#             n=n,
#             )
#     except ValueError:
#         eps_with_zeros_and_ones = "infinity"

#     try:
#         eps_without_zeros_and_ones = compute_epsilon_given_delta_full_range(
#             delta=1e-9, 
#             p=0.5, 
#             n=n, 
#             exclude_all_zeros=True,
#             exclude_all_ones=True,
#             )
#     except:
#         eps_without_zeros_and_ones = "infinity"
#     try:
#         eps_without_zeros = compute_epsilon_given_delta_full_range(
#             delta=1e-9, 
#             p=0.5, 
#             n=n, 
#             exclude_all_zeros=True,
#             exclude_all_ones=False,
#             )
#     except:
#         eps_without_zeros = "infinity"
#     try:
#         eps_without_ones = compute_epsilon_given_delta_full_range(
#             delta=1e-9, 
#             p=0.5, 
#             n=n, 
#             exclude_all_zeros=False,
#             exclude_all_ones=True,
#             )
#     except:
#         eps_without_ones = "infinity"
#     print("n, eps_with_zeros_and_ones : ", n, eps_with_zeros_and_ones, "\n")
#     print("n, eps_without_zeros_and_ones : ", n, eps_without_zeros_and_ones, "\n")
#     print("n, eps_without_zeros : ", n, eps_without_zeros, "\n")
#     print("n, eps_without_ones : ", n, eps_without_ones, "\n")

## Uncomment if you want to see the private ranges given n, p, eps
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
