import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt

"""
Implementation of method outlined in 'Linear Interpolation of Histograms'
In this example, the method is used to interpolate between two gaussians with
sigma=1 and sigma=sqrt(2), to get a gaussian with sigma=1.25
"""

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

def gaussian_cdfinv(sigma_val, y_range):
    cdf_inv = np.sqrt(2)*sigma_val*erfinv(2*y_range-1)
    plt.plot(y_range, cdf_inv, '.')
    return cdf_inv

sigma_min = 1
sigma_max =4
sigma_interp = 3

y_range = np.linspace(0, 1, 101)
cdf_inv1 = gaussian_cdfinv(sigma_min, y_range)
cdf_inv_interp = gaussian_cdfinv(sigma_interp, y_range)
cdf_inv2 = gaussian_cdfinv(sigma_max, y_range)

wt2 = (sigma_interp-sigma_min)/(sigma_max-sigma_min)
wt1 = 1-wt2

inv_interp = wt1*cdf_inv1 + wt2*cdf_inv2

plt.figure()
plt.plot(y_range, inv_interp)

g1 = gaussian(cdf_inv1, sigma=sigma_min)
g2 = gaussian(cdf_inv2, sigma=sigma_max)

g125 = gaussian(cdf_inv_interp, sigma=sigma_interp)

interp = (g1*g2)/(wt1*g2+wt2*g1)

plt.figure()
plt.plot((wt1*cdf_inv1+wt2*cdf_inv2), interp,'.')
#plt.plot(cdf_inv1, g1, 'r')
#plt.plot(cdf_inv2, g2, 'b')
plt.plot(cdf_inv_interp, g125, 'k')
plt.show()