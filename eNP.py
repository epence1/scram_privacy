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
    # "bound_on_x": [],
    # "a": [],
    # "x": [],
    # "ratio": [],
    # "prob_0_given_count": [],
    # "prob_1_given_count": [],
    "probability_of_bad": [],
}


def compute_odds_ratio_doesnt_hold(n, p, bound_on_x):
    # total_odds = 0
    # for c in range(int((n - 1) * p) + int(math.ceil(bound_on_x)), n):
    #     odds = binom.pmf(k=c, n=n - 1, p=p)
    #     total_odds += odds
    # return total_odds
    largest_good_val = int((n - 1) * p) + int(math.ceil(bound_on_x)) - 1
    return 1 - binom.cdf(k=largest_good_val, n=n - 1, p=p)


for n in NUM_PARTICPANTS:
    for p in PROBABILITIES:
        for eps in EPSILON_VALS:
            # Compute bound on x for the ratio to remain bounded by 1+eps < e^eps
            bound_on_x = (p * eps * (1 - p) * (n - 1)) / (1 + p * eps) - (1 - p) / (
                1 + p * eps
            )
            # for a in range(int((n - 1) * p), n):
            #     # Compute deviation from expected value, represented by x
            #     x = a - int(p * (n - 1))  # x should only ever be an int???????

            #     if x < 0:
            #         break

            #     if (n - 1 - x / (1 - p)) == 0:
            #         ratio = None
            #     else:
            #         # Calculate probablity ratio based on binomial representation
            #         ratio = (n - 1 + (x + 1) / p) / (n - 1 - x / (1 - p))

            #     # P[t_i=0|c_t=a] = (n-a)/n; maybe a should be pn??...
            #     prob_0_given_count = (n - a) / n
            #     # P[t_i=1|c_t=a] = a/n; maybe a should actually be p(n-1)+x+1??
            #     prob_1_given_count = (a) / n

            probability_of_bad = compute_odds_ratio_doesnt_hold(n, p, bound_on_x)

            # Create Table
            table["eps"].append(eps)
            table["p"].append(p)
            table["n"].append(n)
            # table["x"].append(x)
            # table["a"].append(a)
            # table["ratio"].append(ratio)
            # table["prob_0_given_count"].append(prob_0_given_count)
            # table["prob_1_given_count"].append(prob_1_given_count)
            table["probability_of_bad"].append(probability_of_bad)
            # table["bound_on_x"].append(bound_on_x)

print(tabulate(table, headers="keys"))
