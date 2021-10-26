import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.signal import convolve
from scipy.misc import derivative

"""
Exploratory work looking at using the CDF of a gaussian as a convolution
kernel and then numerically differentiating to get broadened spectrum
"""

x = np.linspace(-6,6,101)

def create_synthetic_data(npts, n_peaks):      
    # Set up simple test data
    data = np.zeros(npts)

    idx_list = random.sample(range(0,npts), n_peaks)
    peaks = [round(random.uniform(0.1,1.0),2) for i in range(n_peaks)]

    for (i, p) in zip(idx_list, peaks):
        data[i] = p

    # set up bins
    bins = np.linspace(0,100,npts+1)
    bin_mp = (bins[1:]+bins[:-1])/2

    frequencies = bin_mp

    # set sigma values
    sigma = frequencies*0.1+1

    return data, sigma, bins, bin_mp, frequencies

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

def gaussian_cdf(x, sigma=2, center=0):
    cdf = 0.5*(1+erf((x-center)/(np.sqrt(2)*sigma)))
    return cdf


fig, [ax1, ax2] = plt.subplots(nrows=2,
                               sharex=True,
                               gridspec_kw={
                               'height_ratios': [2, 1]})

fig2, ax = plt.subplots()

sigma_min = 1
sigma_max = np.sqrt(2)

for sigma, color in zip(np.linspace(sigma_min, sigma_max, 7),
                            ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']):
    wt2 = (sigma-sigma_min)/(sigma_max-sigma_min)
    wt1 = 1-wt2

    actual = gaussian(x,sigma=sigma)
    actual_cdf = gaussian_cdf(x, sigma=sigma)

    est_cdf = wt1*gaussian_cdf(x,sigma=sigma_min) + wt2*gaussian_cdf(x,sigma=sigma_max)

    def cdf_est(x):
        return wt1*gaussian_cdf(x,sigma=sigma_min) + wt2*gaussian_cdf(x,sigma=sigma_max)
    
    est1 = derivative(cdf_est, x, dx=1e-6)

    lin_interp = wt1*gaussian(x, sigma=sigma_min) + wt2*gaussian(x, sigma=sigma_max)

    x_offset = (0 + ((sigma - sigma_min)
                       * (40 - 0) / (sigma_max - sigma_min)))

    rms = np.sqrt(np.mean((actual-est1)**2))

    diff = est1-lin_interp

    print('diff sum')
    print(sum(diff))

    ax1.plot(x+x_offset, actual, color=color)
    ax1.plot(x+x_offset, est1, '--', color=color)
    ax1.plot(x+x_offset, lin_interp, '.', color=color)
    ax2.plot([x_offset],[rms],'o')
    ax.plot(x, actual_cdf, color=color)
    ax.plot(x, est_cdf, '--', color=color)

"""
Test accuracy of linear interpolation of cdf's
"""

# evaluate gaussian function and gaussian cdf for sigma=1
# and sigma=sqrt(2)
gauss1 = gaussian(x, sigma=1)
cdf1 = gaussian_cdf(x, sigma=1)

gauss2 = gaussian(x, sigma=np.sqrt(2))
cdf2 = gaussian_cdf(x, sigma=np.sqrt(2))

plt.figure()
plt.plot(x,cdf1,label="sigma = 1")
plt.plot(x,cdf2, label="sigma = sqrt(2)")

# estimate cdf for sigma = 1.25 and plot
wt2 = (1.25-1)/(np.sqrt(2)-1)
wt1 = 1-wt2
cdf125 = wt1*cdf1 + wt2*cdf2

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
    cdf1 = gaussian_cdf(x, sigma=1)
    cdf2 = gaussian_cdf(x,sigma=np.sqrt(2))
    wt2 = (1.25-1)/(np.sqrt(2)-1)
    wt1 = 1-wt2
    cdf_interp = wt1*cdf1 + wt2*cdf2
    return convolve(data, cdf_interp, mode="same")

# convolve data with pdf
pdf_conv = convolve(data, gaussian(x,sigma=1.25), mode="same")

# convolve data with cdf then numerically differentiate
cdf_conv_deriv = derivative(fun, x, dx=1e-6, args=(data,))

plt.figure()
plt.plot(bin_mp, cdf_conv_deriv,'b')

plt.plot(bin_mp, pdf_conv,'r--')

plt.show()
