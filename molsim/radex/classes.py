from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..classes import Continuum, Observation, Spectrum
from ..utils import _apply_beam, _apply_aperture

from ..constants import cm, ckm, h, k
from .interface import run as radex_run, read as radex_read


def _planck(T, freq):
    f = freq * 1e6
    return (1e26 * 2 * h / cm**2) * f**3 / np.expm1((h * f) / (k * T))


def _merge_intervals(intervals):
    merged = []
    for x in sorted(intervals, key=lambda x: x[0]):
        if merged and x[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x[1])
        else:
            merged.append(list(x))
    return merged


def _intersect_intervals(a, b):
    intersected = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            intersected.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return intersected


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


@dataclass
class MaserSimulation:
    """Class for maser simulation"""

    spectrum: Spectrum = None           # Spectrum object associated with this simulation
    observation: Observation = None     # Observation object associated with this simulation
    source: List[NonLTESource] = None   # List of NonLTESource objects associated with this simulation
    continuum: Continuum = None         # Continuum object
    size: float = 1e3                   # source size
    ll: List[float] = None              # lower limits [MHz]
    ul: List[float] = None              # upper limits [MHz]
    sim_width: float = 10.0             # FWHMs to simulate +/- line center
    res: float = 0.01                   # resolution if simulating line profiles [MHz]
    use_obs: bool = False               # flag for line profile simulation to be done with observations
    aperture: float = None              # aperture size for spectrum extraction [arcsec]

    def __post_init__(self):
        # set default values
        if self.spectrum is None:
            self.spectrum = Spectrum()

        if self.source is None:
            self.source = NonLTESource()
        # cast to list if needed
        if isinstance(self.source, NonLTESource):
            self.source = [self.source]

        if self.continuum is None:
            self.continuum = Continuum(params=2.725)

        if self.use_obs:
            if self.observation is None:
                raise RuntimeError('use_obs is True but observation is not given')

        # use minimum from observation or -infinity
        if self.ll is None:
            if self.use_obs:
                self.ll = self.observation.spectrum.frequency.min()
            else:
                self.ll = -np.Infinity
        # cast to list if needed
        if isinstance(self.ll, float) or isinstance(self.ll, int):
            self.ll = [self.ll]

        # use maximum from observation or physically impossible large value
        if self.ul is None:
            if self.use_obs:
                self.ul = self.observation.spectrum.frequency.max()
            else:
                self.ul = np.Infinity
        # cast to list if needed
        if isinstance(self.ul, float) or isinstance(self.ul, int):
            self.ul = [self.ul]

        # finished setting default values, the rest is done in _update()
        self._update()

    def _update(self):
        # first, set up the frequency grid
        if self.use_obs:
            frequency = np.copy(self.observation.spectrum.frequency)
        else:
            lines_intervals = _merge_intervals(
                [tuple(freq * (1 + d * self.sim_width * source.dV / ckm) for d in [-1, 1])
                 for source in self.source for freq in source.frequency]
            )
            drawn_intervals = list(zip(self.ll, self.ul))
            combined_intervals = _intersect_intervals(lines_intervals, drawn_intervals)
            if combined_intervals:
                frequency = np.concatenate([np.arange(s, e, self.res) for s, e in combined_intervals])
            else:
                frequency = np.empty(0)
        self.spectrum.freq_profile = frequency

        # second, compute optical depth and intensity for each source, add to the total intensity
        intensity = np.zeros_like(frequency)
        cumulative_tau = np.zeros_like(frequency)
        for source in self.source:
            tau, Iv = source.get_tau_Iv(frequency, self.sim_width)
            intensity += Iv * np.exp(-cumulative_tau)
            cumulative_tau += tau

        # third, compute the contribution from background continuum source, and subtract the continuum
        intensity += self.continuum.Ibg(frequency) * np.expm1(-cumulative_tau)

        # fourth, correct for beam dilution for gaussian beam or uniform aperture
        if self.observation is not None:
            if self.observation.observatory.sd is True:
                intensity, beam_dilution = _apply_beam(frequency, intensity, self.size, self.observation.observatory.dish, return_beam=True)
            if self.observation.observatory.array is True and self.aperture is not None:
                intensity, beam_dilution = _apply_aperture(intensity, self.size, self.aperture, return_beam=True)

        self.spectrum.int_profile = intensity
        self.beam_dilution = beam_dilution
