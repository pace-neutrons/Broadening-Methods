import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g*0.2

margin = 0.05

"""
Compare linear interpolation, optimised interpolation and peak-biased optimisation
"""

def plot_linear_interp():
    """Plot linearly-interpolated Gaussians"""

    g1_center = 0
    g2_center = 40
    sigma_max = 4
    sigma_min = 1

    x = np.linspace(-10, 10, 101)
    npts = 10

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3,
                                        sharex=True,
                                        gridspec_kw={
                                            'height_ratios': [3, 1, 1]})

    for sigma, color in zip(np.linspace(sigma_min, sigma_max, npts),
                            ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8','C9']):
        x_offset = (g1_center
                  + ((sigma - sigma_min)
                     * (g2_center - g1_center) / (sigma_max - sigma_min)))
        actual = gaussian(x, sigma=sigma)
        low_ref = gaussian(x, sigma=sigma_min)
        high_ref = gaussian(x, sigma=sigma_max)
        mix = (sigma - sigma_min) / (sigma_max - sigma_min)
        est = (1 - mix) * low_ref + mix * high_ref
        ax1.plot(x + x_offset, actual, color=color)
        ax1.plot(x + x_offset, est, c=color, linestyle='--')

        rms = np.sqrt(np.mean((actual - est)**2))
        SKL = sum(np.log(actual/est)*actual) + sum(np.log(est/actual)*est)

        ax2.plot([x_offset], [rms], 'o', c='C0')
        ax3.plot([x_offset], [SKL], 'o', c='C0')

plot_linear_interp()

def plot_optimised_interp():

    g1_center = 0
    g2_center = 40
    sigma_max = 4
    sigma_min = 1

    x = np.linspace(-10, 10, 101)
    npts = 10

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3,
                                        sharex=True,
                                        gridspec_kw={
                                        'height_ratios': [3, 1, 1]})

    def gaussian_mix(x, w1, w2):
        """Return a linear combination of two Gaussians with weights"""
        return (w1 * gaussian(x, sigma=sigma_min)
                + w2 * gaussian(x, sigma=sigma_max))


    for sigma, color in zip(np.linspace(sigma_min, sigma_max, npts),
                            ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8','C9']):

        ydata = gaussian(x, sigma=sigma)
        (mix1, mix2), _ = curve_fit(gaussian_mix, x, ydata, p0=[0.5, 0.5])

        x_offset = (g1_center
                    + ((sigma - sigma_min)
                       * (g2_center - g1_center) / (sigma_max - sigma_min)))
        actual = gaussian(x, sigma=sigma)
        est = gaussian_mix(x, mix1, mix2)

        rms = np.sqrt(np.mean((actual - est)**2))
        SKL = sum(np.log(actual/est)*actual) + sum(np.log(est/actual)*est)
        ax1.plot(x + x_offset, actual, color=color)
        ax1.plot(x + x_offset, est, color=color, linestyle='--')

        ax2.plot([x_offset], [rms], 'o', c='C0')
        ax3.plot([x_offset], [SKL], 'o', c='C0')

plot_optimised_interp()

def plot_peak_biased_interp():

    g1_center = 0
    g2_center = 40
    sigma_max = 4
    sigma_min = 1

    x = np.linspace(-10, 10, 101)
    npts = 10

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3,
                                        sharex=True,
                                        gridspec_kw={
                                        'height_ratios': [3, 1, 1]})

    def gaussian_mix(x, w1):
        """Return a linear combination of two Gaussians with weights"""
        return (w1 * gaussian(x, sigma=sigma_min)
                + (1-w1) * gaussian(x, sigma=sigma_max))


    for sigma, color in zip(np.linspace(sigma_min, sigma_max, npts),
                            ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8','C9']):

        ydata = gaussian(x, sigma=sigma)

        uncertainty = np.ones(len(x))
        uncertainty[51] = 1e-10

        mix1, _ = curve_fit(gaussian_mix, x, ydata, p0=[0.5], bounds=(0,1), sigma = uncertainty, absolute_sigma = True)

        mix1 = mix1[0]

        x_offset = (g1_center
                    + ((sigma - sigma_min)
                       * (g2_center - g1_center) / (sigma_max - sigma_min)))
        actual = gaussian(x, sigma=sigma)
        est = gaussian_mix(x, mix1)

        rms = np.sqrt(np.mean((actual - est)**2))
        SKL = sum(np.log(actual/est)*actual) + sum(np.log(est/actual)*est)
        ax1.plot(x + x_offset, actual, color=color)
        ax1.plot(x + x_offset, est, color=color, linestyle='--')

        ax2.plot([x_offset], [rms], 'o', c='C0')
        ax3.plot([x_offset], [SKL], 'o', c='C0')

plot_peak_biased_interp()

plt.show()