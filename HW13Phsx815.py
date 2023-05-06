import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #https://www.tutorialspoint.com/scipy/scipy_stats.htm

# normal distribution 
np.random.seed(0)
data = np.random.normal(loc=5, scale=2, size=1000) #https://www.geeksforgeeks.org/rand-vs-normal-numpy-random-python/

# kernel function
#https://medium.com/@BillChen2k/kernel-density-estimation-with-python-from-scratch-c200b187b6c4
def kernel_function(x, xi, h):
    return norm.pdf((x - xi) / h) 

# data fromn kernel
#https://bookdown.org/egarpor/NP-UC3M/kde-i-kde.html
def kernel(x, data, h):
    n = len(data)
    density = np.zeros_like(x)
    for xi in data:
        density += kernel_function(x, xi, h)
    density /= (n * h)
    return density

# density estimation
x = np.linspace(0, 10, 100)
density = kernel(x, data, h=0.5)

# Plotting data and density
plt.hist(data, bins=30, density=True, alpha=0.5)
plt.plot(x, density)
plt.show()
