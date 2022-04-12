from scipy.stats import binom
from math import exp
from decimal import *


def noiseless_privacy_analysis(n, p, eps):
    def prob_count_appearing(a):
        return binom.pmf(k=a, n=n, p=p)

    # def get_ratio_for_count(a):
    # assert a < n and a > 0
    # numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))
    # denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))
    # # print(
    # #     "frac data: ",
    # #     a,
    # #     round(numerator, 10),
    # #     round(denominator, 10),
    # #     numerator / denominator,
    # # )
    # if denominator == 0:
    #     print("bad inputs", a, p, n, denominator, numerator)
    # return numerator / denominator

    def is_eps_noiseless_private(a):
        if a >= n or a == 0:
            return 0
        thresh = Decimal(exp(eps))

        ## expanded get_ratio
        assert a < n and a > 0
        numerator = Decimal(binom.pmf(k=a, n=n - 1, p=p))
        denominator = Decimal(binom.pmf(k=a - 1, n=n - 1, p=p))
        return numerator < thresh * denominator and denominator < thresh * numerator

        # ratio = get_ratio_for_count(a)
        # if ratio == 0:
        #     print("bad ratio", ratio)
        # return float(ratio < thresh) and (Decimal(1.0) / ratio < thresh)

    exp_val = 0
    is_eps_private_flag = False
    flag_switched = False
    lower_a, upper_a = -1, -1
    for a in range(0, n + 1):
        if not flag_switched:
            if is_eps_noiseless_private(a) and not is_eps_private_flag:
                lower_a = a
                is_eps_private_flag = True
            elif not is_eps_noiseless_private(a) and is_eps_private_flag:
                upper_a = a
                is_eps_private_flag = False
                flag_switched = True
        else:
            ## should be not private
            if is_eps_noiseless_private(a):
                print("going back and forth...")
                print("good a range: [", lower_a, ",", upper_a, ")")
                print("new a", a)
                assert 1 == 0
        exp_val += prob_count_appearing(a) * is_eps_noiseless_private(a)
    print("good a range: [", lower_a, ",", upper_a, ")")
    return exp_val


# n_vals = [10, 100, 1000]
n_vals = [10, 100]
p_vals = [0.1, 0.5, 0.75]
eps_vals = range(1, 6)
# n_vals = [10]
# p_vals = [0.5]
# eps_vals = [0.1]
for n in n_vals:
    for p in p_vals:
        for eps_big in eps_vals:
            # eps = float(eps_big) / 10.0
            eps = float(eps_big)
            delta = noiseless_privacy_analysis(n, p, eps)
            print(n, p, eps, delta, "\n")
