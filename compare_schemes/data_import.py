# import data
from euphonic.cli.utils import (load_data_from_file, _grid_spec_from_args, _get_energy_bins)
from euphonic.util import mp_grid, mode_gradients_to_widths
from euphonic import ureg
import numpy as np
import random


def import_real_data(filename, grid_space, energy_broadening):
    """
    Function that imports data from a given file and processes it using Euphonic
    functionality. Returns an unbroadened dos spectrum for the data, as well as
    sigma values, bin centres adn frequency values which can then be used to
    perform broadening

    :param filename: Phonon data file contatining force constants. File format must
    be one of: .yaml, force_constants.hdf, .castep_bin, .check or .json
    :type filename: str
    :param grid_space: Optional arguemtns to set q-point spacing of mp-grid.
    :type grid_space: float
    :param energy_broadening: Scale factor for multiplying the mode widths for
    adaptive broadening
    :type energy_broadening: float

    :returns:
        - dos - unbroadened dos spectrum
        - sigma - mode widths multiplied by `energy_broadening`
        - bin_mp - bin centres
        - freqs - mode frequencies
    """

    data = load_data_from_file(filename)

    recip_length_unit = ureg('1 /angstrom')
    grid_spec = _grid_spec_from_args(data.crystal, grid=None,
                                        grid_spacing=(grid_space
                                                    * recip_length_unit))

    # for adaptive broadening
    modes, mode_grads = data.calculate_qpoint_phonon_modes(mp_grid(grid_spec), return_mode_gradients=True)
    mode_widths = mode_gradients_to_widths(mode_grads, modes.crystal.cell_vectors)
    mode_widths *= energy_broadening

    modes.frequencies_unit = 'hartree'

    ebins = _get_energy_bins(modes, 4001)

    dos = modes._calculate_dos(ebins)
    sigma = mode_widths.to('hartree').magnitude
    bins = ebins.to('hartree').magnitude
    bin_mp = bins[:-1] + 0.5*np.diff(bins)
    freqs = modes.frequencies.magnitude

    return dos, sigma, bins, bin_mp, freqs

def create_synthetic_data(npts, n_peaks):
    """
    Function that generates synthetic data that can be used to test
    broadening schemes

    :param npts: specifies number of points for the data
    :type npts: float
    :param n_peaks: specifies the number of non-zero peaks in the data
    :type n_peaks: float

    :returns:
        - data - raw, unbroadened data
        - sigma - array of sigma values
        - bin_mp - bin centres
        - freqs - mode frequencies
    """        

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