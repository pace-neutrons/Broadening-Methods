# error relationship
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import *

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

# plot estimated gaussians (sigma=1.25) for various sigma spacing values
sigma_spacing = np.linspace(4,np.sqrt(2),101)
x = np.linspace(-5,5,101)

sigma = 1.25
ydata = gaussian(x, sigma=sigma)

fig, [ax1, ax2] = plt.subplots(nrows=2,
                                        sharex=True,
                                        gridspec_kw={
                                        'height_ratios': [1, 1]})

fig2, ax = plt.subplots()

fig3, ax3 = plt.subplots()

ax1.plot(x, ydata, label='actual')

sigma_min = 1

for space in sigma_spacing:
    sigma_max = space

    def gaussian_mix(x, w1, w2):
        """Return a linear combination of two Gaussians with weights"""
        return (w1 * gaussian(x, sigma=sigma_min)
                + w2 * gaussian(x, sigma=sigma_max))
    
    (mix1, mix2), _ = curve_fit(gaussian_mix, x, ydata, p0=[0.5, 0.5])
    est = gaussian_mix(x, mix1, mix2)

    k = 3*(mix1*sigma_min**4+mix2*sigma_max**4)/(mix1*sigma_min**2+mix2*sigma_max**2)**2 - 3

    diff = (est-ydata)

    #print(sum(diff))

    # find area between actual gaussian and approximation
    w = symbols('w')
    actual_gaussian = exp(-0.5 * (w / sigma)**2) / (sigma * sqrt(2 * pi))
    est_gaussian =  mix1*(exp(-0.5 * (w / sigma_min)**2) / (sigma_min * sqrt(2 * pi)))\
        + mix2*(exp(-0.5 * (w / sigma_max)**2) / (sigma_max * sqrt(2 * pi)))

    test = integrate(actual_gaussian,(w,-oo, oo))

    test2 = integrate(est_gaussian, (w,-oo, oo))

    area = test2-test

    ax.plot([space],[area],'.',color='C1')

    ax3.plot([space],[k],'.',color='C2')

    ax1.plot(x, est, label='sigma=' + str(round(space,4)))
    ax2.plot(x, diff, label='sigma=' + str(round(space,4)))

ax1.legend()
ax2.legend()

plt.show()

# sigma_max = 2
# sigma_vals = np.linspace(sigma_min, sigma_max, 101)

# plt.figure()

# for s in sigma_vals:

#     sigma=s

#     def gaussian_mix(x, w1, w2):
#         """Return a linear combination of two Gaussians with weights"""
#         return (w1 * gaussian(x, sigma=sigma_min)
#                 + w2 * gaussian(x, sigma=sigma_max))

#     ydata = gaussian(x, sigma=sigma)
    
#     (mix1, mix2), _ = curve_fit(gaussian_mix, x, ydata, p0=[0.5, 0.5])
#     est = gaussian_mix(x, mix1, mix2)

#     diff = (est-ydata)/ydata

#     #print(sum(diff))

#     # find area between actual gaussian and approximation
#     w = symbols('w')
#     actual_gaussian = exp(-0.5 * (w / sigma)**2) / (sigma * sqrt(2 * pi))
#     est_gaussian =  mix1*(exp(-0.5 * (w / sigma_min)**2) / (sigma_min * sqrt(2 * pi)))\
#         + mix2*(exp(-0.5 * (w / sigma_max)**2) / (sigma_max * sqrt(2 * pi)))

#     test = integrate(actual_gaussian,(w,-oo, oo))

#     test2 = integrate(est_gaussian, (w,-oo, oo))

#     area = test2-test

#     plt.plot([sigma],[area],'.')

# plt.show()


