# Collection of broadening schemes that can be run with real/synthetic data
import numpy as np
from scipy.signal import convolve

def gaussian_eval(sigma, points, center):
    """
    Function which evaluates a gaussian function with standard
    deviation `sigma` and center `center` at given points.

    :param sigma: Standard deviation of the gaussian kernel, can be a scalar for
    fixed width broadening, or array of values for adaptive broadening
    :type sigma: float or 1D array-like
    :param points: points at which the Gaussian function should be evaluated
    :type points: 1D array
    :param center: Gaussian center
    :type center: float or 1D array

    :returns: array with calculated Gaussian values
    """
    # calculate bin width, which is then used to normalize the Gaussian
    b_width = points[1] - points[0] 
    kernel = np.exp(-(points - center)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))*b_width

    return kernel

def interpolate_scheme(sigma, bins, points, frequencies, data, width_factor):
    """
    Interpolate broadening scheme as it is implemented in Abins
    """

    log_bases = {'2':2,'sqrt2':np.sqrt(2),'1.23':1.23,'1.25':1.25}

    spacing = log_bases[width_factor]

    n_kernels = int(np.ceil(np.log(max(sigma)/min(sigma))/np.log(spacing)))
    sigma_samples = spacing**np.arange(n_kernels+1)*min(sigma)

    bin_width = points[1]-points[0]
    hist = data

    freq_range = 3*max(sigma)
    kernel_npts_oneside = np.ceil(freq_range/bin_width)

    kernels = gaussian_eval(sigma=sigma_samples[:, np.newaxis], points=np.arange(-kernel_npts_oneside, kernel_npts_oneside+1, 1)*bin_width, center=0)

    spectra = np.array([convolve(hist, kernel, mode="same") for kernel in kernels])

    sigma_locations = np.searchsorted(sigma_samples, sigma)

    spectrum = np.zeros_like(points)

    spectrum[sigma_locations==0] = spectra[0, sigma_locations==0]

    mix_functions = {'gaussian': {'1.23': {'lower':[-1.269, 8.016, -17.47, 11.73],
                                           'upper':[1.450, -8.922, 18.82, -11.34]},
                                  '1.25': {'lower':[-1.147, 7.304, -16.06, 10.90],
                                           'upper':[1.324, -8.194, 17.39, -10.52]},
                                  '2': {'lower': [-0.1873, 1.464, -4.079, 3.803],
                                            'upper': [0.2638, -1.968, 5.057, -3.353]},
                                    'sqrt2': {'lower': [-0.6079, 4.101, -9.632, 7.139],
                                                'upper': [0.7533, -4.882, 10.87, -6.746]}}}

    for i in range(1, len(sigma_samples)):
            masked_block = (sigma_locations == i)
            sigma_factors = sigma[masked_block] / sigma_samples[i - 1]
            lower_mix = np.polyval(mix_functions['gaussian'][width_factor]['lower'], sigma_factors)
            upper_mix = np.polyval(mix_functions['gaussian'][width_factor]['upper'], sigma_factors)

            spectrum[masked_block] = (lower_mix * spectra[i-1, masked_block]
                                    + upper_mix * spectra[i, masked_block])

    return spectrum

def interpolate_scheme2(sigma, bins, points, frequencies, data, width_factor):
    """
    Modified version of the interpolate scheme, which scales spectrum with
    pre-computed mixing weights before convolution step
    """

    log_bases = {'2':2,'sqrt2':np.sqrt(2),'1.23':1.23,'1.25':1.25}
    spacing = log_bases[width_factor]

    n_kernels = int(np.ceil(np.log(max(sigma)/min(sigma))/np.log(spacing)))
    sigma_samples = spacing**np.arange(n_kernels+1)*min(sigma)

    bin_width = points[1]-points[0]
    hist = data

    freq_range = 3*max(sigma)
    kernel_npts_oneside = np.ceil(freq_range/bin_width)

    kernels = gaussian_eval(sigma=sigma_samples[:, np.newaxis], points=np.arange(-kernel_npts_oneside, kernel_npts_oneside+1, 1)*bin_width, center=0)

    sigma_locations = np.searchsorted(sigma_samples, sigma)

    mix_functions = {'gaussian': {'1.23': {'lower':[-1.269, 8.016, -17.47, 11.73],
                                           'upper':[1.450, -8.922, 18.82, -11.34]},
                                  '1.25': {'lower':[-1.147, 7.304, -16.06, 10.90],
                                           'upper':[1.324, -8.194, 17.39, -10.52]},
                                  '2': {'lower': [-0.1873, 1.464, -4.079, 3.803],
                                        'upper': [0.2638, -1.968, 5.057, -3.353]},
                                  'sqrt2': {'lower': [-0.6079, 4.101, -9.632, 7.139],
                                            'upper': [0.7533, -4.882, 10.87, -6.746]}}}

    convolved_spectrum = np.zeros((len(frequencies)))

    for i in range(1, len(sigma_samples)):
        masked_block = (sigma_locations == i)
        sigma_factors = sigma[masked_block] / sigma_samples[i - 1]

        lower_mix = np.polyval(mix_functions['gaussian'][width_factor]['lower'], sigma_factors)
        upper_mix = np.polyval(mix_functions['gaussian'][width_factor]['upper'], sigma_factors)

        lower_scaled = np.zeros((len(frequencies)))
        upper_scaled = np.zeros((len(frequencies)))
        lower_scaled[masked_block] = lower_mix*hist[masked_block]
        upper_scaled[masked_block] = upper_mix*hist[masked_block]

        lower_convolve = convolve(lower_scaled, kernels[i-1], mode="same")
        upper_convolve = convolve(upper_scaled, kernels[i], mode="same")

        convolved_spectrum += lower_convolve + upper_convolve
    
    return convolved_spectrum
