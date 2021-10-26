# explore limits for sigma spacing
# i.e. can we evaluate Gaussian at sigma=0 and sigma=inf and 
# interpolate exactly in between these distributions

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

def gaussian_cdf(x, sigma=2, center=0):
    cdf = 0.5*(1+erf((x-center)/(np.sqrt(2)*sigma)))
    return cdf

# find limit as sigma->0
x = np.linspace(0,1, 101)

gauss_eval = gaussian(x,sigma=10e-9)

#plt.plot(x, gauss_eval)

cdf_eval = gaussian_cdf(x,sigma=10e-9)

# heaviside step function - cdf when sigma=0
H = np.heaviside(x,0.5)

plt.figure()
plt.plot(x, cdf_eval)
plt.plot(x, H)

# find limit as sigma->inf
cdf_eval2 = gaussian_cdf(x,sigma=10e9)

cdf_lim = 1/2*np.ones(len(x))

plt.figure()
plt.plot(x, cdf_eval2)
plt.figure()
plt.plot(x, cdf_lim)
plt.show()
