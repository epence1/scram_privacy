from eNP_direct import ENPPrivacy
from deniable_privacy import DeniablePrivacy
import matplotlib.pyplot as plt
import numpy as np

# NMIN = 3
# NMAX = 100
# n_vals = range(NMIN, NMAX)
# enp_eps_vals = []
# dp_eps_vals = []
# enp_prob_0_n_for_n = []
# PROB = 0.75
# DELTA = 1e-9
# for n in n_vals:
#     print(n)
#     enp = ENPPrivacy(n=n, p=PROB)
#     dp = DeniablePrivacy(n=n, p=PROB)
#     enp_eps, enp_range = enp.get_min_eps_slow(failure_rate=DELTA)
#     dp_eps, dp_range = dp.get_min_eps_slow(failure_rate=DELTA)
#     print("ENP MIN EPS SLOW (eps, output_range): ", enp_eps, enp_range)
#     print("DENP MIN EPS SLOW (eps, output_range): ", dp_eps, dp_range)
#     enp_eps_vals.append(enp_eps)
#     dp_eps_vals.append(dp_eps)
    # enp_prob_0 = enp.prob_output_appearing(0)
    # enp_prob_1 = enp.prob_output_appearing(n)
    # print(enp_prob_0+enp_prob_1)
    # enp_prob_0_n_for_n.append((enp_prob_0+enp_prob_1))


# ## Probability Mass of Combined count=0,n for various n
# plt.plot(n_vals, enp_prob_0_n_for_n, marker="o", label="eNP: Combined Probability Mass of Count=0 and Count=n")
# plt.title("Combined Probability Mass of Count=0 and Count=n for Various n: P="+str(PROB))
# plt.axhline(y = 1e-9, color = 'r', linestyle = '-', label="Delta=1e-9")
# plt.xlabel("n")
# plt.ylabel("Log(Probability)")
# plt.xticks(np.arange(NMIN, NMAX+1, 5.0))
# plt.yscale("log")
# #plt.ylim(-1,10)
# plt.legend()
# plt.show()


# ## Minimum epsilon comparison between enp and dp
# plt.plot(n_vals, enp_eps_vals, marker="o", label="eNP: minimum epsilon for delta=10^-9")
# plt.plot(n_vals, dp_eps_vals, marker="x", label="Deniable: minimum epsilon for delta=10^-9")
# plt.title("Minimum Achievable Epsilon for Various N: P="+str(PROB))
# plt.xlabel("n")
# plt.ylabel("epsilon")
# plt.ylim(-1,5)
# plt.legend()
# plt.show()

# enp_max_eps = []
# dp_max_eps = []
# for n in n_vals:
#     print(n)
#     enp_max = 1e-10
#     dp_max = 1e-10
#     for p in np.arange(0.2,0.8,0.1):
#         enp = ENPPrivacy(n=n, p=PROB)
#         dp = DeniablePrivacy(n=n, p=PROB)
#         enp_eps, enp_range = enp.get_min_eps_slow(failure_rate=DELTA)
#         dp_eps, dp_range = dp.get_min_eps_slow(failure_rate=DELTA)
#         enp_max = max(enp_max, enp_eps)
#         dp_max = max(dp_max, dp_eps)
#     enp_max_eps.append(enp_max)
#     dp_max_eps.append(dp_max)

# print(enp_max_eps)
# print(dp_max_eps)
# plt.plot(n_vals, enp_max_eps, marker="o", label="eNP: min epsilon for delta=10^-9")
# plt.plot(n_vals, dp_max_eps, marker="x", label="Deniable: minimum epsilon for delta=10^-9")
# plt.title("Minimum Achievable Epsilon for ANY p in 0.1 to 0.9")
# plt.xlabel("n")
# plt.ylabel("epsilon")
# plt.ylim(-1,5)
# plt.legend()
# plt.show()

## Plot Probability Density for Deniable Privacy
N = 20
den_priv = DeniablePrivacy(n=N, p=0.5)
enp_priv = ENPPrivacy(n=N, p=0.5)
enp_probability_density_x = []
enp_probability_density_y = []
dp_probability_density_x = []
dp_probability_density_y = []

for a in range(0,N+1):
    enp_probability_density_x.append(a)
    dp_probability_density_x.append(a)
    enp_probability_density_y.append(enp_priv.prob_output_appearing(a))
    if a==0 or a==N:
        dp_probability_density_y.append(0)
    else:
        dp_probability_density_y.append(den_priv.prob_output_appearing(a))

plt.plot(enp_probability_density_x, enp_probability_density_y, marker="o", label="eNP Probability Density")
plt.plot(dp_probability_density_x, dp_probability_density_y, marker="x", label="Deniable Probability Density")
plt.xticks(np.arange(min(enp_probability_density_x), 2.0, 1.0))

plt.title("eNP vs Deniable: Probability Density Function of Possible Outputs for N="+str(N))
plt.xlabel("count=a")
plt.ylabel("Log(Probability of seeing output count=a)")
plt.yscale("log")
plt.legend()
plt.show()

print(enp_probability_density_y)
print(dp_probability_density_y)


