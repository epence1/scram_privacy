from tabulate import tabulate
EPSILON_VALS = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9]
PROBABILITIES= [0.1, 0.25, 0.5, 0.75, 0.9]
NUM_PARTICPANTS = [1, 10, 100, 1000, 10000, 100000]

table = {"eps": [], "p": [], "n": [], "x" : [], "ratio" : []}

for eps in EPSILON_VALS:
    for p in PROBABILITIES:
        for n in NUM_PARTICPANTS:
            # Set x to the max value it can be while the ratio remains bounded by 1+eps < e^eps
            x = (p*eps*(1-p)*(n-1))/(1+p*eps) - (1-p)/(1+p*eps)
            # properly_bounded = (x < (n-1)*(1-p)) # I think this is implied by the above condition...
            ratio = (n-1+(x+1)/p)/(n-1-x/(1-p))
            table["eps"].append(eps)
            table["p"].append(p)
            table["n"].append(n)
            table["x"].append(x)
            table["ratio"].append(ratio)

print(tabulate(table, headers='keys'))

