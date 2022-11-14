from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from functools import lru_cache, partial
import math
from typing import Any, Callable, Dict, FrozenSet, List, Optional, TextIO, Tuple, Type, Union

import numpy as np
from scipy.interpolate import interp1d

from ..classes import Continuum, Observation, Spectrum
from ..utils import _apply_beam, _apply_aperture

from ..constants import cm, ckm, h, k


def _planck(T, freq):
    f = freq * 1e6
    return (1e26 * 2 * h / cm**2) * f**3 / np.expm1((h * f) / (k * T))


def _rayleigh_jeans_temperature(Iv, freq):
    f = freq * 1e6
    return (1e-26 * cm**2 / (2 * k)) * Iv / f**2


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


@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=True)
class Levels:
    num_levels: int
    level_numbers: np.ndarray[int]
    level_energies: np.ndarray[float]
    statistical_weight: np.ndarray[float]
    quantum_numbers: np.ndarray[str]
    level_number_index_map: Dict[int, int]

    def __repr__(self) -> str:
        return f'<Levels object with {len(self.level_numbers)} levels>'

    @classmethod
    def from_LAMDA(cls: Type[Levels], file_in: TextIO) -> Levels:
        fields: Dict[str, Any] = dict()

        _ = file_in.readline()
        fields['num_levels'] = int(file_in.readline().split(maxsplit=1)[0])

        _ = file_in.readline()
        fields['level_numbers'] = list()
        fields['level_energies'] = list()
        fields['statistical_weight'] = list()
        fields['quantum_numbers'] = list()
        for _ in range(fields['num_levels']):
            cords = file_in.readline().split(maxsplit=4)
            fields['level_numbers'].append(int(cords[0]))
            fields['level_energies'].append(float(cords[1]))
            fields['statistical_weight'].append(float(cords[2]))
            fields['quantum_numbers'].append(cords[3])
        fields['level_numbers'] = np.array(fields['level_numbers'])
        fields['level_energies'] = np.array(fields['level_energies'])
        fields['statistical_weight'] = np.array(fields['statistical_weight'])
        fields['quantum_numbers'] = np.array(fields['quantum_numbers'])

        fields['level_number_index_map'] = dict()
        for index, level_number in enumerate(fields['level_numbers']):
            fields['level_number_index_map'][level_number] = index

        return cls(**fields)


@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=True)
class RadiativeTransitions:
    num_transitions: int
    transition_numbers: np.ndarray[int]
    upper_level_numbers: np.ndarray[int]
    lower_level_numbers: np.ndarray[int]
    spontaneous_decay_rates: np.ndarray[float]
    upper_level_indices: np.ndarray[int]
    lower_level_indices: np.ndarray[int]

    def __repr__(self:RadiativeTransitions) -> str:
        return f'<RadiativeTransitions object with {len(self.transition_numbers)} transitions>'

    @classmethod
    def from_LAMDA(cls: Type[RadiativeTransitions], file_in: TextIO, levels: Levels) -> RadiativeTransitions:
        fields: Dict[str, Any] = dict()

        _ = file_in.readline()
        fields['num_transitions'] = int(file_in.readline().split(maxsplit=1)[0])

        _ = file_in.readline()
        fields['transition_numbers'] = list()
        fields['upper_level_numbers'] = list()
        fields['lower_level_numbers'] = list()
        fields['spontaneous_decay_rates'] = list()
        for i in range(fields['num_transitions']):
            cords = file_in.readline().split(maxsplit=4)
            fields['transition_numbers'].append(int(cords[0]))
            fields['upper_level_numbers'].append(int(cords[1]))
            fields['lower_level_numbers'].append(int(cords[2]))
            fields['spontaneous_decay_rates'].append(float(cords[3]))
        fields['transition_numbers'] = np.array(fields['transition_numbers'])
        fields['upper_level_numbers'] = np.array(fields['upper_level_numbers'])
        fields['lower_level_numbers'] = np.array(fields['lower_level_numbers'])
        fields['spontaneous_decay_rates'] = np.array(fields['spontaneous_decay_rates'])

        def level_number_index_map_func(
            level_number): return levels.level_number_index_map[level_number]
        fields['upper_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['upper_level_numbers'])])
        fields['lower_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['lower_level_numbers'])])

        return cls(**fields)


@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=True)
class CollisionalTransitions:
    num_temperatures: int
    num_transitions: int
    temperatures: np.ndarray[float]
    transition_numbers: np.ndarray[int]
    upper_level_numbers: np.ndarray[int]
    lower_level_numbers: np.ndarray[int]
    rate_coefficients: np.ndarray[float]
    upper_level_indices: np.ndarray[int]
    lower_level_indices: np.ndarray[int]

    def __repr__(self) -> str:
        return f'<CollisionalTransitions object with {len(self.transition_numbers)} transitions at {len(self.temperatures)} temperatures>'

    @classmethod
    def from_LAMDA(cls: Type[CollisionalTransitions], file_in: TextIO, levels: Levels) -> CollisionalTransitions:
        fields: Dict[str, Any] = dict()

        _ = file_in.readline()
        fields['num_transitions'] = int(file_in.readline().split(maxsplit=1)[0])
        _ = file_in.readline()
        fields['num_temperatures'] = int(file_in.readline().split(maxsplit=1)[0])

        _ = file_in.readline()
        fields['temperatures'] = [*map(float, file_in.readline().split())]

        _ = file_in.readline()
        fields['transition_numbers'] = list()
        fields['upper_level_numbers'] = list()
        fields['lower_level_numbers'] = list()
        fields['rate_coefficients'] = list()
        for i in range(fields['num_transitions']):
            cords = file_in.readline().split()
            fields['transition_numbers'].append(int(cords[0]))
            fields['upper_level_numbers'].append(int(cords[1]))
            fields['lower_level_numbers'].append(int(cords[2]))
            fields['rate_coefficients'].append([*map(float, cords[3:])])
        fields['transition_numbers'] = np.array(fields['transition_numbers'])
        fields['upper_level_numbers'] = np.array(fields['upper_level_numbers'])
        fields['lower_level_numbers'] = np.array(fields['lower_level_numbers'])
        fields['rate_coefficients'] = np.array(fields['rate_coefficients'])

        def level_number_index_map_func(level_number):
            return levels.level_number_index_map[level_number]
        fields['upper_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['upper_level_numbers'])])
        fields['lower_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['lower_level_numbers'])])

        return cls(**fields)


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=True)
class NonLTEMolecule:
    name: str
    weight: float
    levels: Levels
    radiative_transitions: RadiativeTransitions
    collisional_transitions: Dict[str, CollisionalTransitions]
    partner_name_standardizer: Dict[str, str] = field(init=False, repr=False)
    collisional_rate_coefficients_getter: Dict[str, Callable[[
        np.ndarray], np.ndarray]] = field(init=False, repr=False)

    interpolator: InitVar[Union[str, Callable[[np.ndarray,
                                               np.ndarray], Callable[[float], np.ndarray]]]] = 'slinear'

    def __post_init__(self, interpolator: Union[str, Callable[[np.ndarray, np.ndarray], Callable[[float], np.ndarray]]]) -> None:
        if isinstance(interpolator, str):
            interpolator = partial(interp1d, kind=interpolator,
                                   copy=False, assume_sorted=True)
        collisional_rate_coefficients_getter: Dict[str, Callable[[float], np.ndarray]] = {
            partner_name: lru_cache(interpolator(
                collision_data.temperatures, collision_data.rate_coefficients))
            for partner_name, collision_data in self.collisional_transitions.items()
        }
        object.__setattr__(self, 'collisional_rate_coefficients_getter',
                           collisional_rate_coefficients_getter)

        partner_name_variants = {
            'H2':  ['h2'],
            'pH2': ['ph2', 'p-H2', 'p-h2', 'para-H2', 'para-h2'],
            'oH2': ['oh2', 'o-H2', 'o-h2', 'ortho-H2', 'ortho-h2'],
            'e':   ['e-', 'electron', 'electrons'],
            'H':   ['h'],
            'He':  ['he'],
            'H+':  ['h+']
        }
        partner_name_standardizer: Dict[str, str] = {
            variant_name: standard_name
            for standard_name, variant_names in partner_name_variants.items()
            for variant_name in variant_names
        }
        object.__setattr__(self, 'partner_name_standardizer',
                           partner_name_standardizer)

    @classmethod
    def from_LAMDA(cls: Type[NonLTEMolecule], file_in: Union[str, TextIO], **kwargs) -> NonLTEMolecule:
        fields: Dict[str, Any] = dict(kwargs)

        if isinstance(file_in, str):
            file_in = open(file_in, 'r')

        """
        link: https://home.strw.leidenuniv.nl/~moldata/molformat.html

        Format of LAMDA datafiles
        Below follows a description of the format adopted for presenting the atomic and molecular data in LAMDA. Any similarities with datafiles from other authors is completely coincidental.

        Lines 1 - 2: molecule (or atom) name
        Lines 3 - 4: molecular (or atomic) weight (a.m.u.)
        Lines 5 - 6: number of energy levels (NLEV)
        Lines 7 - 7+NLEV: level number, level energy (cm-1), statistical weight. These numbers may be followed by additional info such as the quantum numbers, which are however not used by the program. The levels must be listed in order of increasing energy.
        Lines 8+NLEV - 9+NLEV: number of radiative transitions (NLIN)
        Lines 10+NLEV - 10+NLEV+NLIN: transition number, upper level, lower level, spontaneous decay rate (s-1). These numbers may be followed by additional info such as the line frequency, which is however not used by the program.
        Lines 11+NLEV+NLIN - 12+NLEV+NLIN: number of collision partners (NPART)
        This is followed by NPART blocks of collision data:
        Lines 13+NLEV+NLIN - 14+NLEV+NLIN: collision partner ID and reference. Valid identifications are: 1=H2, 2=para-H2, 3=ortho-H2, 4=electrons, 5=H, 6=He, 7=H+.
        Lines 15+NLEV+NLIN - 16+NLEV+NLIN: number of transitions for which collisional data exist (NCOL)
        Lines 17+NLEV+NLIN - 18+NLEV+NLIN: number of temperatures for which collisional data are given
        Lines 19+NLEV+NLIN - 20+NLEV+NLIN: the NTEMP values of the temperature for which collisional data are given
        Lines 21+NLEV+NLIN - 21+NLEV+NLIN+NCOL: transition number, upper level, lower level; rate coefficients (cm3s-1) at each temperature.
        """
        _ = file_in.readline()
        fields['name'] = file_in.readline().rstrip()

        _ = file_in.readline()
        fields['weight'] = float(file_in.readline().split(maxsplit=1)[0])

        fields['levels'] = Levels.from_LAMDA(file_in)

        fields['radiative_transitions'] = RadiativeTransitions.from_LAMDA(
            file_in, fields['levels'])

        LAMDA_partner_id_name_map = {
            1: 'H2',
            2: 'pH2',
            3: 'oH2',
            4: 'e',
            5: 'H',
            6: 'He',
            7: 'H+'
        }

        _ = file_in.readline()
        num_partners = int(file_in.readline().split(maxsplit=1)[0])
        fields['collisional_transitions'] = dict()
        for i in range(num_partners):
            _ = file_in.readline()
            partner_id = int(file_in.readline().split(maxsplit=1)[0])
            if partner_id not in LAMDA_partner_id_name_map:
                raise ValueError(f'Unregconized collision partner ID: {partner_id = }')

            partner_name = LAMDA_partner_id_name_map[partner_id]
            fields['collisional_transitions'][partner_name] = CollisionalTransitions.from_LAMDA(
                file_in, fields['levels'])

        return cls(**fields)


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

    spectrum: Optional[Spectrum] = None             # Spectrum object associated with this simulation
    observation: Optional[Observation] = None       # Observation object associated with this simulation
    source: Optional[List[NonLTESource]] = None     # List of NonLTESource objects associated with this simulation
    continuum: Optional[Continuum] = None           # Continuum object
    size: Optional[float] = 1e3                     # source size
    ll: Optional[List[float]] = None                # lower limits [MHz]
    ul: Optional[List[float]] = None                # upper limits [MHz]
    sim_width: float = 10.0                         # FWHMs to simulate +/- line center
    res: float = 0.01                               # resolution if simulating line profiles [MHz]
    units: str = 'K'                                # units for the simulation; accepts 'K', 'mK', 'Jy/beam'
    use_obs: bool = False                           # flag for line profile simulation to be done with observations
    aperture: Optional[float] = None                # aperture size for spectrum extraction [arcsec]

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

        # fifth, set units
        if self.units in ['K', 'mK']:
            intensity = _rayleigh_jeans_temperature(intensity, frequency)
            if self.units == 'mK':
                intensity *= 1e3
        if self.units in ['Jy/beam', 'Jy']:
            omega = self.observation.observatory.synth_beam[0]*self.observation.observatory.synth_beam[1]
            sr_per_beam = omega * np.pi / (206265**2 * 4 * np.log(2))
            intensity *= sr_per_beam

        self.spectrum.int_profile = intensity
        self.beam_dilution = beam_dilution
