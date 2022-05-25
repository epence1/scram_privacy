from k_deniable_privacy import KDeniablePrivacy
from deniable_privacy import DeniablePrivacy
import matplotlib.pyplot as plt
import numpy as np

K = 4
NMIN = 2*K+1
NMAX = 35
n_vals = range(NMIN, NMAX)
k2_dp_eps_vals = []
k3_dp_eps_vals = []
k4_dp_eps_vals = []
kdp_eps_vals = []
dp_eps_vals = []
PROB = 0.9
DELTA = 1e-9
for n in n_vals:
    print(n)
    k2_dp = KDeniablePrivacy(n=n, p=PROB, k=2)
    k3_dp = KDeniablePrivacy(n=n, p=PROB, k=3)
    k4_dp = KDeniablePrivacy(n=n, p=PROB, k=4)
    dp = DeniablePrivacy(n=n, p=PROB)
    k2_dp_eps, k2_dp_range = k2_dp.get_min_eps_slow(failure_rate=DELTA)
    k3_dp_eps, k3_dp_range = k3_dp.get_min_eps_slow(failure_rate=DELTA)
    k4_dp_eps, k4_dp_range = k4_dp.get_min_eps_slow(failure_rate=DELTA)
    dp_eps, dp_range = dp.get_min_eps_slow(failure_rate=DELTA)
    print("k2DP MIN EPS SLOW (eps, output_range): ", k2_dp_eps, k2_dp_range)
    print("k3DP MIN EPS SLOW (eps, output_range): ", k3_dp_eps, k3_dp_range)
    print("k4DP MIN EPS SLOW (eps, output_range): ", k4_dp_eps, k4_dp_range)
    print("DENP MIN EPS SLOW (eps, output_range): ", dp_eps, dp_range)
    k2_dp_eps_vals.append(k2_dp_eps)
    k3_dp_eps_vals.append(k3_dp_eps)
    k4_dp_eps_vals.append(k4_dp_eps)
    dp_eps_vals.append(dp_eps)


## Minimum epsilon comparison between kdp and dp
plt.plot(n_vals, k2_dp_eps_vals, marker="o", label="2-Deniable: minimum epsilon for delta=10^-9")
plt.plot(n_vals, k3_dp_eps_vals, marker="x", label="3-Deniable: minimum epsilon for delta=10^-9")
plt.plot(n_vals, k4_dp_eps_vals, marker="o", label="4-Deniable: minimum epsilon for delta=10^-9")
plt.plot(n_vals, dp_eps_vals, marker="x", label="Deniable: minimum epsilon for delta=10^-9")
plt.title("Minimum Achievable Epsilon for Various N: P=" + str(PROB))# + ", K=" + str(K))
plt.xlabel("n")
plt.ylabel("epsilon")
plt.ylim(-1,5)
plt.legend()
plt.show()
