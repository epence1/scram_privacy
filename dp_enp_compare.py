from eNP_direct import ENPPrivacy
from deniable_privacy import DeniablePrivacy

PROB = 0.5
DELTA = 1e-3
for n in range(3, 30):
    print(n)
    enp = ENPPrivacy(n=n, p=PROB)
    denp = DeniablePrivacy(n=n, p=PROB)
    print("ENP MIN EPS SLOW (eps, output_range): ", enp.get_min_eps_slow(failure_rate=DELTA))
    print("DENP MIN EPS SLOW (eps, output_range): ", denp.get_min_eps_slow(failure_rate=DELTA))

