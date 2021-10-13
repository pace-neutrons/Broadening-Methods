# run broadening on real data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from data_import import import_real_data
from broadening_schemes import interpolate_scheme, interpolate_scheme2

def broaden_real(filename, grid_space=0.1, energy_broadening=1.5):
    """
    Broaden data output from import_real_data and plot results, comparing different methods
    and sigma_spacing
    """

    data, sigma, bins, bin_mp, frequencies = import_real_data(filename, grid_space, energy_broadening)
    n_modes = frequencies.shape[-1]
    weights = np.full(len(frequencies), 1/len(frequencies))
    mode_weights_calc = np.full((len(frequencies), n_modes), 1/(len(frequencies)*n_modes))

    # evalulate gaussian function at each point - reference method
    result_gauss = np.zeros(len(bin_mp))
    for q in range(len(frequencies)):
        for m in range(n_modes):
            kernel = norm.pdf(bin_mp, loc=frequencies[q,m], scale=sigma[q,m])
            result_gauss += kernel*weights[q]

    # get frequency values from bin centres
    freqs = bin_mp

    sigma_vals = np.zeros(len(freqs))
    # get sigma values associated with frequencies
    for i, f in enumerate(freqs):
        idx = (np.abs(np.ravel(frequencies)-f)).argmin()
        sigma_vals[i] = np.ravel(sigma)[idx]

    # ensure no zero values to avoid divide by zero error
    sigma_vals[sigma_vals == 0] = 10**-8

    spectrum = interpolate_scheme(sigma_vals, None, bin_mp, freqs, data, width_factor='sqrt2')
    convolved_spectrum = interpolate_scheme2(sigma_vals, None, bin_mp, freqs, data, width_factor='sqrt2')

    spectrum_123 = interpolate_scheme(sigma_vals, None, bin_mp, freqs, data, width_factor='1.23')
    convolved_spectrum_123 = interpolate_scheme2(sigma_vals, None, bin_mp, freqs, data, width_factor='1.23')

    spectrum_125 = interpolate_scheme(sigma_vals, None, bin_mp, freqs, data, width_factor='1.25')
    convolved_spectrum_125 = interpolate_scheme2(sigma_vals, None, bin_mp, freqs, data, width_factor='1.25')

    # plot unbroadened spectrum
    plt.figure()
    plt.plot(bin_mp, data)

    plt.figure()
    plt.plot(bin_mp, result_gauss, label="Gaussian")
    plt.plot(bin_mp, spectrum, label="Interpolate, spacring sqrt2")
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