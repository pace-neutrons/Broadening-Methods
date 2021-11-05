# look at optimising interpolation parameters by minimizing SKL divergence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

sigma_min = 1
sigma_max = np.sqrt(2)
sigma = 1.25

x = np.linspace(-5,5,101)
actual = gaussian(x, sigma=sigma)

def SKL_weights(x, sigma):
    w_array = np.linspace(0,1,1001)
    actual = gaussian(x, sigma=sigma)
    SKL_save = np.zeros(len(w_array))

    for i,wt1 in enumerate(w_array):
        wt2 = 1-wt1

        est = wt1*gaussian(x, sigma=sigma_min) + wt2*gaussian(x,sigma=sigma_max)

        SKL = sum(np.log(actual/est)*actual) + sum(np.log(est/actual)*est)

        SKL_save[i] = SKL

    idx = np.argmin(SKL_save)

    sw1 = w_array[idx]
    sw2 = 1-w_array[idx]

    # print('SKL Optimization')
    # print(w_array[idx])
    # print(1-w_array[idx])
    return sw1, sw2

# linear interpolation weights
def lin_interp_weights(x, sigma):
    wt2 = (sigma - sigma_min) / (sigma_max - sigma_min)
    wt1 = 1-wt2

    # print('linear interpolation')
    # print(wt1)
    # print(wt2)
    return wt1, wt2

# least squares optimised interpolation
def least_sqaures_weights(x, sigma):

    def gaussian_mix(x, wt):
        """Return a linear combination of two Gaussians with weights"""
        return (wt * gaussian(x, sigma=sigma_min)
                + (1-wt) * gaussian(x, sigma=sigma_max))

    actual = gaussian(x, sigma=sigma)

    mix, _ = curve_fit(gaussian_mix, x, actual, p0=[0.5], bounds=(0,1))

    mix1 = mix
    mix2 = 1-mix

    # print('least squares')
    # print(mix1)
    # print(mix2)
    return mix1, mix2

sigma_vals = np.linspace(sigma_min, sigma_max, 101)

SKL_fit = np.zeros(len(sigma_vals))
lin_fit = np.zeros(len(sigma_vals))
ls_fit = np.zeros(len(sigma_vals))

SKL_rms = np.zeros(len(sigma_vals))
lin_rms = np.zeros(len(sigma_vals))
ls_rms = np.zeros(len(sigma_vals))

g1 = gaussian(x, sigma=sigma_min)
g2 = gaussian(x, sigma=sigma_max)

for i, sigma in enumerate(sigma_vals):
    s1, s2 = SKL_weights(x, sigma)
    l1, l2 = lin_interp_weights(x, sigma)
    ls1, ls2 = least_sqaures_weights(x, sigma)

    SKL_mix = s1*g1 + s2*g2
    lin_mix = l1*g1 + l2*g2
    ls_mix = ls1*g1 + ls2*g2

    actual = gaussian(x, sigma=sigma)

    SKL_fit[i] = sum(np.log(actual/SKL_mix)*actual) + sum(np.log(SKL_mix/actual)*SKL_mix)
    lin_fit[i] = sum(np.log(actual/lin_mix)*actual) + sum(np.log(lin_mix/actual)*lin_mix)
    ls_fit[i] = sum(np.log(actual/ls_mix)*actual) + sum(np.log(ls_mix/actual)*ls_mix)

    SKL_rms[i] = np.sqrt(np.mean((actual - SKL_mix)**2))
    lin_rms[i] = np.sqrt(np.mean((actual - lin_mix)**2))
    ls_rms[i] = np.sqrt(np.mean((actual - ls_mix)**2))

plt.figure()
plt.plot(sigma_vals, SKL_fit, 'k', label='SKL Optimised')
plt.plot(sigma_vals, lin_fit, 'b', label='linear interpolation')
plt.plot(sigma_vals, ls_fit , 'r', label='Least-squares optimised')
plt.legend()

plt.figure()
plt.plot(sigma_vals, SKL_rms, 'k', label='SKL Optimised')
plt.plot(sigma_vals, lin_rms, 'b', label='linear interpolation')
plt.plot(sigma_vals, ls_rms, 'r', label='Least-squares optimised')
plt.legend()

plt.figure()
plt.plot(x, actual,'--')
plt.plot(x, s1*g1+s2*g2, 'k')
plt.plot(x, l1*g1 + l2*g2, 'b')
plt.plot(x, ls1*g1+ls2*g2, 'r')

plt.show()