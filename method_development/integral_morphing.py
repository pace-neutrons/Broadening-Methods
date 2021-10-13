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

plt.figure()
y_range = np.linspace(0, 1, 101)
cdf_inv1 = gaussian_cdfinv(1, y_range)
cdf_inv125 = gaussian_cdfinv(1.25, y_range)
cdf_invr2 = gaussian_cdfinv(np.sqrt(2), y_range)

wt2 = (1.25-1)/(np.sqrt(2)-1)
wt1 = 1-wt2

inv_interp = wt1*cdf_inv1 + wt2*cdf_invr2

plt.plot(y_range, inv_interp)

g1 = gaussian(cdf_inv1, sigma=1)
g2 = gaussian(cdf_invr2, sigma=np.sqrt(2))

g125 = gaussian(cdf_inv125, sigma=1.25)

interp = (g1*g2)/(wt1*g2+wt2*g1)

plt.figure()
plt.plot((wt1*cdf_inv1+wt2*cdf_invr2), interp,'.')
plt.plot(cdf_inv1, g1, 'r')
plt.plot(cdf_invr2, g2, 'b')
plt.plot(cdf_inv125, g125, 'k')
plt.show()