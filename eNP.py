from tabulate import tabulate
import math
EPSILON_VALS = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9]
PROBABILITIES= [0.1, 0.25, 0.5, 0.75, 0.9]
NUM_PARTICPANTS = [1, 10, 100, 1000,]

table = {"eps": [], 
            "p": [], 
            "n": [], 
            "x" : [], 
            "ratio" : [], 
            "prob_0_given_count" : [], 
            "prob_1_given_count" : [], 
            "chernoff" : [],
            }

for eps in EPSILON_VALS:
    for p in PROBABILITIES:
        for n in NUM_PARTICPANTS:
            # Set x to the max value it can be while the ratio remains bounded by 1+eps < e^eps
            x = (p*eps*(1-p)*(n-1))/(1+p*eps) - (1-p)/(1+p*eps)
            
            # properly_bounded = (x < (n-1)*(1-p)) # I think this is implied by the above condition...
            ratio = (n-1+(x+1)/p)/(n-1-x/(1-p))
           
            # P[t_i=0|c_t=a]; count is p(n-1)+x?? ...or just pn...
            prob_0_given_count = (n-(p*(n-1)+x))/n
            # P[t_i=1|c_t=a]; count here is actually p(n-1)+x+1??
            prob_1_given_count = (p*(n-1)+x)/n
           
            # Compute Chernoff Bound(Pr[X>=A])
            m = (n-1)*(1-p) # Max value that X can take on: n'-pn'=n'*(1-p)=(n-1)*(1-p)
            t = 1
            mgf_x = ((math.e**t)*p+1-p)**m # moment generating function for Binomial RV X
            exponentiated_A = math.e**(t*(x))
            chernoff = mgf_x/exponentiated_A
            
            # Create Table
            table["eps"].append(eps)
            table["p"].append(p)
            table["n"].append(n)
            table["x"].append(x)
            table["ratio"].append(ratio)
            table["prob_0_given_count"].append(prob_0_given_count)
            table["prob_1_given_count"].append(prob_1_given_count)
            table["chernoff"].append(chernoff)

print(tabulate(table, headers='keys'))

