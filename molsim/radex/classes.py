from dataclasses import dataclass, field

import numpy as np

from ..constants import cm, ckm, h, k
from .interface import run as radex_run, read as radex_read


def _planck(T, freq):
    f = freq * 1e6
    return (1e26 * 2 * h / cm**2) * f**3 / np.expm1((h * f) / (k * T))


@dataclass
class NonLTESource:
    """Class for keeping non-LTE source properties"""

    velocity: float = 0.0   # lsr velocity [km/s]
    dV: float = 3.0         # FWHM [km/s]
    column: float = 1e10    # column density [cm^-2]
    Tbg: float = 2.725      # background temperature [K]
    Tkin: float = 30.0      # kinetic temperature [K]
    # density of the collision partner as a dictionary [cm^-3]
    collision_density: dict = field(default_factory=lambda: {'H2': 1e4})
    # LAMDA collisional data file
    collision_file: str = 'hco+.dat'
    # temperary file that store output from RADEX
    radex_output: str = '/tmp/radex.out'

    def __post_init__(self):
        # run RADEX simulation to obtain excitation temperatures and optical depths
        outfile = radex_run(
            molfile=self.collision_file,
            outfile=self.radex_output,
            f_low=3.0,  # 3 GHz is the minimum since RADEX cannot display wavelength > 1e5 um
            f_high=1e5, # 1e5 GHz is the maximum since RADEX cannot display frequency > 1e5 GHz
            T_k=self.Tkin,
            n_c=self.collision_density,
            T_bg=self.Tbg,
            N=self.column,
            dV=self.dV
        )
        parameters_keys, parameters_values, grid = radex_read(outfile)
        # RADEX precision is low, choose smartly between wavelength and frequency
        # TODO: use collision_file to correct the frequency
        self.frequency = np.array([cm / g['WAVEL'] if g['WAVEL'] > g['FREQ'] else g['FREQ'] * 1e3 for g in grid])
        self.frequency *= 1 - self.velocity / ckm
        # T_EX and TAU may be insensitive to small changes in input
        # reimplementing RADEX is required to correct the problem
        self.Tex = np.array([g['T_EX'] for g in grid])
        self.tau = np.array([g['TAU'] for g in grid])

    def get_tau_Iv(self, freq, sim_width = 10.0):
        tau = np.zeros_like(freq)
        Iv = np.zeros_like(freq)

        for freq_, Tex_, tau_ in zip(self.frequency, self.Tex, self.tau):
            dfreq = self.dV / ckm * freq_
            two_sigma_sq = dfreq**2 / (4 * np.log(2))
            lo = np.searchsorted(freq, freq_ - sim_width * dfreq, side='left')
            hi = np.searchsorted(freq, freq_ + sim_width * dfreq, side='right')
            f = freq[lo:hi]
            t = tau_ * np.exp(-(f - freq_)**2 / two_sigma_sq)
            tau[lo:hi] += t
            Iv[lo:hi] += _planck(Tex_, f) * t  # TODO: double-check this expression

        mask = tau > 0.0
        Iv[mask] *= -np.expm1(-tau[mask]) / tau[mask]
        return tau, Iv
