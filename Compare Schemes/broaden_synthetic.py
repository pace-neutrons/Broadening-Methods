# run broadening on synthetic data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from data_import import create_synthetic_data
from broadening_schemes import gaussian_eval, interpolate_scheme, interpolate_scheme2

def broaden_synthetic(npts=1000, npeaks=10):
    """ 
    Broaden synthetic data and plot results to compare different methods
    """

    data, sigma, bins, bin_mp, frequencies = create_synthetic_data(npts, npeaks)

    # reference method
    kernels = gaussian_eval(sigma[:,np.newaxis], bin_mp, frequencies[:,np.newaxis])
    result_gauss = np.dot(data, kernels)

    # interpolate broadening method
    spectrum = interpolate_scheme(sigma, bins, bin_mp, frequencies, data, width_factor='sqrt2')
    convolved_spectrum = interpolate_scheme2(sigma, bins, bin_mp, frequencies, data, width_factor='sqrt2')

    spectrum_123 = interpolate_scheme(sigma, bins, bin_mp, frequencies, data, width_factor='1.23')
    convolved_spectrum_123 = interpolate_scheme2(sigma, bins, bin_mp, frequencies, data, width_factor='1.23')

    spectrum_125 = interpolate_scheme(sigma, bins, bin_mp, frequencies, data, width_factor='1.25')
    convolved_spectrum_125 = interpolate_scheme2(sigma, bins, bin_mp, frequencies, data, width_factor='1.25')

    # plot unbroadened data
    plt.figure()
    plt.plot(bin_mp, data)

    plt.figure()
    plt.plot(bin_mp, result_gauss, label="Gaussian")
    plt.plot(bin_mp, spectrum, label="Interpolate, spacing sqrt2")
    plt.plot(bin_mp, spectrum_123, label="Interpolate, spacing 1.23")
    plt.plot(bin_mp, spectrum_125, label="Interpolate, spacing 1.25")
    plt.legend()

    plt.figure()
    plt.plot(bin_mp, result_gauss, label="Gaussian")
    plt.plot(bin_mp, convolved_spectrum, label="Interpolate (modified), spacring sqrt2")
    plt.plot(bin_mp, convolved_spectrum_123, label="Interpolate (modified), spacing 1.23")
    plt.plot(bin_mp, convolved_spectrum_125, label="Interpolate (modified), spacing 1.25")
    plt.legend()

    plt.show()