import numpy as np
from euphonic.cli.utils import (load_data_from_file, _grid_spec_from_args, _get_energy_bins)
from euphonic.util import mp_grid, mode_gradients_to_widths
from euphonic import ureg
import matplotlib.pyplot as plt

"""
Script that produces violin plots of the sigma values associated with different data sets.
Can be used to determine whether to calculate sigma samples over whole sigma range, or whether
to skip over some values.
"""

#filenames = ['mp-147-20180417.yaml', 'mp-200-20180417.yaml', 'mp-216-20180417.yaml', 'mp-226-20180417.yaml',
#             'mp-306-20180417.yaml', 'mp-661-20180417.yaml', 'mp-830-20180417.yaml', 'mp-917-20180417.yaml',
#             'mp-1818-20180417.yaml', 'mp-2319-20180417.yaml', 'mp-7041-20180417.yaml']

filenames = ['mp-661-20180417.yaml', 'mp-917-20180417.yaml']

for filename in filenames:
    data = load_data_from_file('/home/jessfarmer/Broadening_Methods/Data/'+filename)

    recip_length_unit = ureg('1 /angstrom')
    grid_spec = _grid_spec_from_args(data.crystal, grid=None,
                                        grid_spacing=(0.1
                                                    * recip_length_unit))

    # for adaptive broadening
    modes, mode_grads = data.calculate_qpoint_phonon_modes(mp_grid(grid_spec), return_mode_gradients=True)
    mode_widths = mode_gradients_to_widths(mode_grads, modes.crystal.cell_vectors)

    modes.frequencies_unit = 'hartree'

    sigma = mode_widths.to('hartree').magnitude

    plt.figure()
    plt.violinplot(np.ravel(sigma))

plt.show()