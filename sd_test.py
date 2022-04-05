import random
import matplotlib.pyplot as plt
import statistics      

LENGTH_INPUT = 100
LOWER_BOUND = 1
UPPER_BOUND = 100
NUM_OUTLIERS = 4
OUTLIER_MULTIPLIER = 2
SD_THRESHOLD = 2

input = []

#Generate data
for i in range(LENGTH_INPUT-NUM_OUTLIERS):
    input.append(random.randint(LOWER_BOUND, UPPER_BOUND))

#Insert outliers
for i in range(NUM_OUTLIERS):
    input.append(random.randint(LOWER_BOUND, UPPER_BOUND*OUTLIER_MULTIPLIER))

#Compute SD and mean
sd = statistics.pstdev(input)
mean = sum(input)/LENGTH_INPUT

#Alternate SD computation (used by SCRAM)
sum_of_squares = 0
for i in input:
    sum_of_squares += i*i

square_of_sum = sum(input)**2

sd_alt = ((sum_of_squares-(square_of_sum/LENGTH_INPUT))/LENGTH_INPUT)**(1/2)

#Establish threshold
threshold = mean + SD_THRESHOLD*sd

#Apply threshold
output = []
outliers = []
for x in input:
    if x > threshold:
        outliers.append(x)
    else:
        output.append(x)

#Identify outliers
print("mean: " + str(mean))
print("sd: " + str(sd))
print("sd_alt: " + str(sd_alt))
print("outliers: " + str(outliers))

#Plot
plt.hist(input, bins=UPPER_BOUND)
plt.show()

