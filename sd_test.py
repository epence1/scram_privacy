import random
import matplotlib.pyplot as plt
import statistics      

N = 100
LOWER_BOUND = 1
UPPER_BOUND = 100
NUM_OUTLIERS = 4
OUTLIER_MULTIPLIER = 2
SD_THRESHOLD = 2

def generate_dataset(size=N-NUM_OUTLIERS, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    '''
    Generates dataset of size N-NUM_OUTLIERS so that outliers can be added to dataset

    Args:
    size <int> : nnumber of datapoints to generate
    lower_bound <int> : smallest value a datapoint can take
    upper_bound <int> : largest value a datapoint can take

    Returns:
    data <list> : A list of <int>
    '''
    data = []
    for i in range(size):
        data.append(random.randint(lower_bound, upper_bound))
    return data

def generate_outliers(size=NUM_OUTLIERS, lower_bound=UPPER_BOUND, upper_bound=UPPER_BOUND*OUTLIER_MULTIPLIER):
    '''
    Generates a list of outlier datapoints

    Args:
    size <int> : nnumber of datapoints to generate
    lower_bound <int> : smallest value an outlier can take, which is upper bound of regular dataset
    upper_bound <int> : largest value an outlier can take

    Returns:
    outliers <list> : list of integer outliers
    '''
    outliers = []
    for i in range(size):
        outliers.append(random.randint(lower_bound, upper_bound))
    return outliers

def compute_sd(data):
    '''
    Computes standard deviation of dataset

    Args:
    data <list> : list of integer datapoints

    Returns:
    sd <float> : standard deviation of dataset
    '''
    # #Alternate SD computation (used by SCRAM)
    # sum_of_squares = 0
    # for i in input:
    #     sum_of_squares += i*i

    # square_of_sum = sum(input)**2

    # sd_alt = ((sum_of_squares-(square_of_sum/LENGTH_INPUT))/LENGTH_INPUT)**(1/2)
    
    sd = statistics.pstdev(data)
    return sd

def compute_mean(data, length_data=N):
    '''
    Computes mean of dataset

    Args:
    data <list> : list of integer datapoints
    length_data <integer> : number of datapoints in data (known constant)

    Returns:
    mean <float> : average of data
    '''
    mean = sum(data)/length_data
    return mean

def compute_threshold(mean, sd, threshold_multiplier=SD_THRESHOLD):
    '''
    Computes the distance from the mean beyond which datapoints are labeled as outliers

    Args:
    mean <float> : mean of data
    sd <float> : dtandard deviation of data
    threshold_multiplier <float> : acceptable number of SDs from the mean

    Returns:
    threshold <float> 
    '''
    threshold = mean + threshold_multiplier*sd
    return threshold

def eliminate_outliers(data, threshold):
    '''
    Removes outlier values above the threshold from the dataset

    Args:

    Returns:
    <tuple> : a tuple of lists: the data without outliers, the detected outliers
    '''
    # TODO: Make Recursive!!! Compute threshold for each new version of data set, if outliers remove, if not return
    
    outliers_free = []
    outliers = []
    for x in data:
        if x > threshold:
            outliers.append(x)
        else:
            outliers_free.append(x)
    return outliers_free, outliers

data = generate_dataset()
outliers = generate_outliers()
print("Generated Data: ", data)
print("Generated Outliers: ", outliers)

full_data = data+outliers

mean = compute_mean(full_data)
sd = compute_sd(full_data)
print("Mean of Generated Data: " + str(mean))
print("SD of Generated Data: " + str(sd))

threshold = compute_threshold(mean, sd)
print("SD Threshold of Generated Data: ", threshold)
no_outliers = eliminate_outliers(full_data, threshold)

#Identify outliers
print("no_outliers: " + str(no_outliers))

#Plot
plt.hist(data, bins=UPPER_BOUND)
plt.show()

