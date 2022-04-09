from tabulate import tabulate
from scipy.stats import binom
import math

EPSILON_VALS = [0.001, 0.01, 0.1, 0.2, 0.5]
PROBABILITIES = [0.1, 0.5, 0.9]
NUM_PARTICPANTS = [1, 10, 100, 1000, 10000, 1000000]

table = {
    "n": [],
    "p": [],
    "eps": [],
    "probability_eNP_fails": [],
    "deviation_ceiling": [],
}

def compute_bounded_deviation(eps, p, n):
    '''
    Computes the x value for which the ratio Pr[B = p*n' + x] / 
    Pr[B = p*n' + x + 1] == 1+eps
    (This x-value is significant because all smaller x values ensure
    with provable probaility that the above ratio is bounded by 1+eps)

    Args:
    eps <float> : epsilon parameter
    p <float> : probability of t_i=1 (for any t_i in input T)
    n <int> : number of inputs, or participants

    Returns:
    bound <float>
    '''

    bound = ( (eps * p * (n - 1) * (1 - p)) / (1 + p * eps) ) - ( (1 - p) / (
        1 + p * eps) )
    return bound

def compute_binom_ratio(p, n, x):
    '''
    Computes the ratio Pr[B = p*n' + x] / Pr[B = p*n' + x + 1] where B is Binomial(n'=n-1, p)
    These two probabilities represent two adjacent points, count=a and count=a-1
    Here, the numerator corresponds to count=a-1 and the denominator 
    corresponds to count=a

    Args:
    p <float> : probability of t_i=1 (for any t_i in input T)
    n <int> : number of inputs, or participants
    x <int> : deviation from binomial's expected value

    Returns:
    ratio <float>
    '''

    ratio = (n - 1 + (x + 1) / p) / (n - 1 - (x) / (1 - p))
    return ratio

def compute_prob_input_given_count(count, n):
    '''
    Computes the probabilities of t_i=0 and t_i=1 given the output count

    Args:
    count <int> : number of 1s in input
    n <int> : number of inputs, or participants
    
    Returns:
    prob_0_given_count, prob_1_given_count <tuple> : P[t_i=0|c_t=a], P[t_i=1|c_t=a]
    '''
    
    # P[t_i=0|c_t=a] = (n-a)/n
    prob_0_given_count = (n - count) / n
    # P[t_i=1|c_t=a] = a/n
    prob_1_given_count = (count) / n
    return prob_0_given_count, prob_1_given_count


def compute_prob_eNP_fails(n, p, bound_on_deviation):
    '''
    Computes the probability that deviation x is not within the specified bound, 
    which translates to the probability eNP fails

    Args:
    n <int> : number of inputs, or participants
    p <float> : probability of t_i=1 (for any t_i in input T)
    bound_on_deviation <float> : bound on x to ensure (with provable probability) the binomial ratio < 1+eps

    Returns:
    failure_odds <float> : the probability eNP fails
    '''
    n_prime = n-1
    expected_value_of_binomial = int((n_prime) * p)
    
    # Leo, is this what you meant by over/under approximation; we can either start just below or just above the bound on deviation
    # Alternative could be lowest_bad_count, which == largest_good_count + 1?
    # THIS ASSUMES THE CASE B= n' * p + x since we are setting it at expected_value + A; could be problematic!!!
    largest_good_count = expected_value_of_binomial + int(math.ceil(bound_on_deviation)) - 1
    
    # Cumulative Distribution Function (cdf) computes the probability random variable x is <= largest_good_count (k)
    # We want odds x is greater than largest_good_count, so we do 1 - cdf
    failure_odds = 1 - binom.cdf(k=largest_good_count, n=n_prime, p=p)
    return failure_odds


for n in NUM_PARTICPANTS:
    for p in PROBABILITIES:
        for eps in EPSILON_VALS:

            bounded_deviation = compute_bounded_deviation(eps=eps, p=p, n=n)

            probability_eNP_fails = compute_prob_eNP_fails(n, p, bounded_deviation)

            # Placeholder for computing probability t_i=0,1 given count=a

            # Placeholder for solving for smallest epsilon given a failure rate

            # Placeholder for solving for minimizing failure rate

            # Placeholder for identifying the smallest n for which "delta" (the odds of failure) is reasonable
            # Perhaps grow the range of x, let x be anything that keeps total count valid, if x is positive, if x is negative

            # Update Table
            table["eps"].append(eps)
            table["p"].append(p)
            table["n"].append(n)
            table["probability_eNP_fails"].append(probability_eNP_fails)
            table["deviation_ceiling"].append(bounded_deviation)

print(tabulate(table, headers="keys"))
