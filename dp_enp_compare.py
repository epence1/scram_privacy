from eNP_direct import ENPPrivacy
from deniable_privacy import DeniablePrivacy
import matplotlib.pyplot as plt

NMIN = 3
NMAX = 61
n_vals = range(NMIN, NMAX)
enp_eps_vals = []
dp_eps_vals = []
enp_unprotected_counts = []
dp_unprotected_counts = []
PROB = 0.5
DELTA = 1e-9
for n in n_vals:
    print(n)
    enp = ENPPrivacy(n=n, p=PROB)
    dp = DeniablePrivacy(n=n, p=PROB)
    enp_eps, enp_range = enp.get_min_eps_slow(failure_rate=DELTA)
    dp_eps, dp_range = dp.get_min_eps_slow(failure_rate=DELTA)
    print("ENP MIN EPS SLOW (eps, output_range): ", enp_eps, enp_range)
    print("DENP MIN EPS SLOW (eps, output_range): ", dp_eps, dp_range)
    enp_eps_vals.append(enp_eps)
    dp_eps_vals.append(dp_eps)

    enp_unprotected = enp_range[0] + n-enp_range[1]
    dp_unprotected = dp_range[0] + n-dp_range[1]
    enp_unprotected_counts.append(enp_unprotected)
    dp_unprotected_counts.append(dp_unprotected)

# ## Minimum epsilon comparison between enp and dp
# plt.plot(n_vals, enp_eps_vals, marker="o", label="e-Noiseless: minimum epsilon for delta=10^-9")
# plt.plot(n_vals, dp_eps_vals, marker="x", label="Deniable: minimum epsilon for delta=10^-9")
# plt.title("Minimum Achievable Epsilon")
# plt.xlabel("n")
# plt.ylabel("epsilon")
# plt.ylim(-1,5)
# plt.legend()
# plt.show()

# ## Unprotected Count comparison between enp and dp
# plt.plot(n_vals, enp_unprotected_counts, marker="o", label="e-Noiseless: # unprotected outputs for delta=10^-9")
# plt.plot(n_vals, dp_unprotected_counts, marker="x", label="Deniable: # unprotected outputs for delta=10^-9")
# plt.title("Number of Unprotected Outputs for Various N: eNP vs DP")
# plt.xlabel("n")
# plt.ylabel("epsilon")
# plt.legend()
# plt.show()
