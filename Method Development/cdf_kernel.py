import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from data_import import create_synthetic_data
from scipy.signal import convolve
from scipy.misc import derivative

"""
Exploratory work looking at using the CDF of a gaussian as a convolution
kernel and then numerically differentiating to get broadened spectrum
"""

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

def gaussian_cdf(x, sigma=2, center=0):
    cdf = 0.5*(1+erf((x-center)/(np.sqrt(2)*sigma)))
    return cdf

"""
Test accuracy of linear interpolation of cdf's (needs to be more rigorous)
"""

# evaluate gaussian function and gaussian cdf for sigma=1
# and sigma=sqrt(2)
x = np.linspace(-6,6,101)
gauss1 = gaussian(x, sigma=1)
cdf1 = gaussian_cdf(x, sigma=1)

gaussr2 = gaussian(x, sigma=np.sqrt(2))
cdfr2 = gaussian_cdf(x, sigma=np.sqrt(2))

plt.figure()
plt.plot(x,cdf1,label="sigma = 1")
plt.plot(x,cdfr2, label="sigma = sqrt(2)")

# estimate cdf for sigma = 1.25 and plot
wt2 = (1.25-1)/(np.sqrt(2)-1)
wt1 = 1-wt2
cdf125 = wt1*cdf1 + wt2*cdfr2

plt.plot(x, cdf125, label="sigma = 1.25 (Lin. Interp.)")

# plot true pdf for sigma=1.25
act = gaussian_cdf(x,sigma=1.25)
plt.plot(x,act,'--', label="sigma = 1.25 (True)")
plt.legend()

"""
Test that convolving with cdf then differentiating is equivalent
to convolving with pdf. (Fixed width broadening case) 
"""
data, sigma, bins, bin_mp, frequencies = create_synthetic_data(1000, 5)

# plot data
plt.figure()
plt.plot(bin_mp, data)

# Define function to be numerically differentiated
def fun(x, data):
    cdf = gaussian_cdf(x, sigma=1)
    return convolve(data, cdf, mode="same")

# convolve data with pdf
pdf_conv = convolve(data, gauss1, mode="same")

# convolve data with cdf then numerically differentiate
cdf_conv_deriv = derivative(fun, x, dx=1e-6, args=(data,))

plt.figure()
plt.plot(bin_mp, cdf_conv_deriv,'b')

plt.plot(bin_mp, pdf_conv,'r--')

plt.show()
