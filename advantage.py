from numpy import exp
from scipy.stats import binom
from noise import LaplaceNoise
from decimal import *
DELTA = 10e-9
N = 100
# N = 30
# P = 0.3
# MIN_EPS = 0.1
# MAX_EPS = 3

MIN_P = 0.25
MAX_P = 0.75
P_STEP = 0.05

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

# Epsilon to be modulated
p = MIN_P
while p <= MAX_P:
    expected_correct = 0
    for c in range(0,N):
        # print("Count: ", c)

        # Calculate Pr[C=c | t = 1], Pr[C=c | t = 0]
        prob_c_given_0 = binom.pmf(k=c, n= N - 1, p=p) # fixes nth input as 0
        prob_c_given_1 = binom.pmf(k=c - 1, n= N - 1, p=p)  # fixes nth input as 1

        #print("(prob_c_given_0, prob_c_given_1): ", prob_c_given_0, prob_c_given_1)

        # b (MLE) = max of these two probabilities 
        if prob_c_given_1 < prob_c_given_0:
            b = 0
        else:
            b = 1

        success_rate = compute_prob_input_given_count(c, N)[b]
        # print("Bit Guessed: ", b)
        # Calculate Pr[t=b | C=c] -- probability that adversary is correct
        # prob_correct_list.append(compute_prob_input_given_count(c, N)[b])
        expected_correct += binom.pmf(k=c, n=N, p=p) * success_rate
        #print("Prob Guess is Correct: ", compute_prob_input_given_count(c, N)[b])


    #after finish iterating through c take expectation of adversary's correctness
    # expected_correct = sum(prob_correct_list)/len(prob_correct_list)
    print("(p, Expected Correctness, advantage, max_success_rate): ", p, expected_correct, expected_correct - max(p, 1-p))
    p += P_STEP

            

#so we have epsilon, expected correctness relationship

#and then graph relationship 
#expecting a positive relationship 


#do the same thing for deniable privacy input distribution

#visualize all 3 for each given epsilon
#using what n's? probably small n's because these are the relevant ones to deniable privacy