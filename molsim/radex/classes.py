from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from functools import lru_cache, partial
import math
from typing import Any, Callable, Dict, FrozenSet, List, Optional, TextIO, Tuple, Type, Union

import numba as nb
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from .utils import ClosestLRUCache
from ..classes import Continuum, Observation, Spectrum
from ..utils import _apply_beam, _apply_aperture

from ..constants import ccm, cm, ckm, h, k


def _planck(T, freq):
    f = freq * 1e6
    return (1e26 * 2 * h / cm**2) * f**3 / np.expm1((h * f) / (k * T))


def _rayleigh_jeans_temperature(Iv, freq):
    f = freq * 1e6
    return (1e-26 * cm**2 / (2 * k)) * Iv / f**2


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for x in sorted(intervals, key=lambda x: x[0]):
        if merged and x[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x[1]))
        else:
            merged.append(x)
    return merged


def _intersect_intervals(list_a: Iterable[Tuple[float, float]], list_b: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intersected: List[Tuple[float, float]] = []
    it_a = iter(list_a)
    it_b = iter(list_b)
    a = next(it_a, None)
    b = next(it_b, None)
    while a is not None and b is not None:
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        if lo <= hi:
            intersected.append((lo, hi))
        if a[1] < b[1]:
            a = next(it_a, None)
        else:
            b = next(it_b, None)
    return intersected


def _find_in_intervals(values: npt.NDArray[np.float_], intervals: Iterable[Tuple[float, float]]) -> npt.NDArray[np.int_]:
    indices: List[int] = []
    i = 0
    for left, right in intervals:
        while i < values.size and values[i] <= right:
            if values[i] >= left:
                indices.append(i)
            i += 1
    return np.array(indices)


@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=True)
class Levels:
    num_levels: int
    level_numbers: npt.NDArray[np.int_]
    level_energies: npt.NDArray[np.float_]
    statistical_weight: npt.NDArray[np.float_]
    quantum_numbers: npt.NDArray[np.str_]
    level_number_index_map: Dict[int, int]
    ediff: npt.NDArray[np.float_] = field(init=False)
    gratio: npt.NDArray[np.float_] = field(init=False)

    def __post_init__(self: Levels):
        ediff = self.level_energies[:, None] - self.level_energies[None, :]
        object.__setattr__(self, 'ediff', ediff)

        gratio = self.statistical_weight[:, None] / self.statistical_weight[None, :]
        object.__setattr__(self, 'gratio', gratio)

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
    transition_numbers: npt.NDArray[np.int_]
    upper_level_numbers: npt.NDArray[np.int_]
    lower_level_numbers: npt.NDArray[np.int_]
    spontaneous_decay_rates: npt.NDArray[np.float_]
    frequencies: npt.NDArray[np.float_]
    upper_level_energies: npt.NDArray[np.float_]
    upper_level_indices: npt.NDArray[np.int_]
    lower_level_indices: npt.NDArray[np.int_]
    ediff: npt.NDArray[np.float_]
    gratio: npt.NDArray[np.float_]

    def __post_init__(self: RadiativeTransitions):
        isort = np.argsort(self.frequencies)
        object.__setattr__(self, 'transition_numbers', self.transition_numbers[isort])
        object.__setattr__(self, 'upper_level_numbers', self.upper_level_numbers[isort])
        object.__setattr__(self, 'lower_level_numbers', self.lower_level_numbers[isort])
        object.__setattr__(self, 'spontaneous_decay_rates',
                           self.spontaneous_decay_rates[isort])
        object.__setattr__(self, 'frequencies', self.frequencies[isort])
        object.__setattr__(self, 'upper_level_energies',
                           self.upper_level_energies[isort])
        object.__setattr__(self, 'upper_level_indices', self.upper_level_indices[isort])
        object.__setattr__(self, 'lower_level_indices', self.lower_level_indices[isort])
        object.__setattr__(self, 'ediff', self.ediff[isort])
        object.__setattr__(self, 'gratio', self.gratio[isort])

    def __repr__(self: RadiativeTransitions) -> str:
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
        fields['frequencies'] = list()
        fields['upper_level_energies'] = list()
        for i in range(fields['num_transitions']):
            cords = file_in.readline().split(maxsplit=6)
            fields['transition_numbers'].append(int(cords[0]))
            fields['upper_level_numbers'].append(int(cords[1]))
            fields['lower_level_numbers'].append(int(cords[2]))
            fields['spontaneous_decay_rates'].append(float(cords[3]))
            fields['frequencies'].append(float(cords[4]) * 1e3)
            fields['upper_level_energies'].append(float(cords[5]))
        fields['transition_numbers'] = np.array(fields['transition_numbers'])
        fields['upper_level_numbers'] = np.array(fields['upper_level_numbers'])
        fields['lower_level_numbers'] = np.array(fields['lower_level_numbers'])
        fields['spontaneous_decay_rates'] = np.array(fields['spontaneous_decay_rates'])
        fields['frequencies'] = np.array(fields['frequencies'])
        fields['upper_level_energies'] = np.array(fields['upper_level_energies'])

        def level_number_index_map_func(
            level_number): return levels.level_number_index_map[level_number]
        fields['upper_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['upper_level_numbers'])])
        fields['lower_level_indices'] = np.array([
            *map(level_number_index_map_func, fields['lower_level_numbers'])])

        fields['ediff'] = levels.ediff[fields['upper_level_indices'],
                                       fields['lower_level_indices']]
        fields['gratio'] = levels.gratio[fields['upper_level_indices'],
                                         fields['lower_level_indices']]

        return cls(**fields)

    @lru_cache
    def get_Bul_J(self: RadiativeTransitions, background: Union[Continuum, float]) -> npt.NDArray[np.float_]:
        if isinstance(background, Continuum):
            freq = ccm * self.ediff * 1e-6
            Tbg = background.Tbg(freq)
        else:
            Tbg = background
        return self.spontaneous_decay_rates / np.expm1((h * ccm / k) / Tbg * self.ediff)

    @lru_cache
    def get_Blu_J(self: RadiativeTransitions, background: Union[Continuum, float]) -> npt.NDArray[np.float_]:
        return self.gratio * self.get_Bul_J(background)


@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=True)
class CollisionalTransitions:
    num_temperatures: int
    num_transitions: int
    temperatures: npt.NDArray[np.float_]
    transition_numbers: npt.NDArray[np.int_]
    upper_level_numbers: npt.NDArray[np.int_]
    lower_level_numbers: npt.NDArray[np.int_]
    rate_coefficients: npt.NDArray[np.float_]
    upper_level_indices: npt.NDArray[np.int_]
    lower_level_indices: npt.NDArray[np.int_]

    def __repr__(self) -> str:
        return f'<CollisionalTransitions object with {len(self.transition_numbers)} transitions at {len(self.temperatures)} temperatures>'

    @classmethod
    def from_LAMDA(cls: Type[CollisionalTransitions], file_in: TextIO, levels: Levels) -> CollisionalTransitions:
        fields: Dict[str, Any] = dict()

        _ = file_in.readline()
        fields['num_transitions'] = int(file_in.readline().split(maxsplit=1)[0])
        _ = file_in.readline()
        fields['num_temperatures'] = int(file_in.readline().split(maxsplit=1)[0])
        fields['num_temperatures'] += 1

        _ = file_in.readline()
        fields['temperatures'] = [*map(float, file_in.readline().split())]
        fields['temperatures'].insert(0, 0.0)
        fields['temperatures'] = np.array(fields['temperatures'])

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
            fields['rate_coefficients'][-1].insert(0, 0.0)
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
        float], npt.NDArray]] = field(init=False, repr=False)

    interpolator: InitVar[Union[str, Callable[[npt.NDArray,
                                               npt.NDArray], Callable[[np.float_], npt.NDArray]]]] = 'slinear'

    def __post_init__(self, interpolator: Union[str, Callable[[npt.NDArray, npt.NDArray], Callable[[np.float_], npt.NDArray]]]) -> None:
        if isinstance(interpolator, str):
            interpolator = partial(interp1d, kind=interpolator,
                                   copy=False, assume_sorted=True)
        collisional_rate_coefficients_getter: Dict[str, Callable[[float], npt.NDArray]] = {
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
        partner_name_standardizer.update({
            standard_name: standard_name
            for standard_name in partner_name_variants.keys()
        })
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


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=True)
class EscapeProbability:
    type: str
    set_probability: Callable[[int, npt.NDArray[np.float_],
                               npt.NDArray[np.float_]], None] = field(init=False)

    def __post_init__(self: EscapeProbability):
        if self.type.lower() in ['uniform', 'uniform sphere', 'uniformsphere']:
            @nb.jit
            def func(num_transitions: int, tau: npt.NDArray[np.float_], beta: npt.NDArray[np.float_]):
                c1 = -3.0 / 8.0
                c2 = -4.0 / 15.0
                c3 = -5.0 / 24.0
                c4 = -6.0 / 35.0
                for i in range(num_transitions):
                    t = tau[i]
                    if abs(t) < 0.1:
                        beta[i] = 1.0 + c1 * t * \
                            (1.0 + c2 * t * (1.0 + c3 * t * (1.0 + c4 * t)))
                    elif t > 20.0:
                        ti = 1.0 / t
                        tisq = ti * ti
                        beta[i] = 3.0 * ti * (0.5 - tisq)
                    else:
                        ti = 1.0 / t
                        tisq = ti * ti
                        beta[i] = 3.0 * ti * (0.5 - tisq + (ti + tisq) * np.exp(-t))
        elif self.type.lower() in ['lvg', 'expanding', 'expanding sphere', 'expanding sphere']:
            @nb.jit
            def func(num_transitions: int, tau: npt.NDArray[np.float_], beta: npt.NDArray[np.float_]):
                a = 1.1719833618734954
                c1 = -a / 2.0
                c2 = -a / 3.0
                c3 = -a / 4.0
                c4 = -a / 5.0
                sqrt_four_pi_inv = 1.0 / np.sqrt(4.0 * np.pi)
                for i in range(num_transitions):
                    t = tau[i]
                    if abs(t) < 0.1 / a:
                        beta[i] = 1.0 + c1 * t * \
                            (1.0 + c2 * t * (1.0 + c3 * t * (1.0 + c4 * t)))
                    elif t > 14.0:
                        beta[i] = 1 / (t * np.sqrt(np.log(t * sqrt_four_pi_inv)))
                    else:
                        beta[i] = -np.expm1(-a * t) / (a * t)
        elif self.type.lower() in ['slab']:
            @nb.jit
            def func(num_transitions: int, tau: npt.NDArray[np.float_], beta: npt.NDArray[np.float_]):
                a = 3.0
                c1 = -a / 2.0
                c2 = -a / 3.0
                c3 = -a / 4.0
                c4 = -a / 5.0
                for i in range(num_transitions):
                    t = tau[i]
                    if abs(t) < 0.1 / a:
                        beta[i] = 1.0 + c1 * t * \
                            (1.0 + c2 * t * (1.0 + c3 * t * (1.0 + c4 * t)))
                    else:
                        beta[i] = -np.expm1(-a * t) / (a * t)
        else:
            raise ValueError(f'Unexpected type = {self.type}')

        object.__setattr__(self, 'set_probability', func)


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class NonLTESourceMutableParameters:
    Tkin: float
    collision_density: Dict[str, float]
    background: Union[Continuum, float]
    escape_probability: EscapeProbability
    column: float
    dV: float
    velocity: float

    _mutated: bool = field(init=False, repr=False)

    def __setattr__(self: NonLTESourceMutableParameters, name: str, value: Any):
        untracked = ['_mutated', 'velocity']
        if name not in untracked and (not hasattr(self, name) or getattr(self, name) != value):
            self._mutated = True
        object.__setattr__(self, name, value)

    def is_mutated(self: NonLTESourceMutableParameters, reset: bool = True) -> bool:
        ret = self._mutated
        if reset:
            self._mutated = False
        return ret


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=True)
class NonLTESource:
    molecule: NonLTEMolecule
    mutable_params: NonLTESourceMutableParameters

    miniter: int = 10
    maxiter: int = 10000
    ccrit: float = 1e-6
    Tex_relaxation_coefficient: float = 0.5
    xpop_relaxation_coefficient: float = 0.3
    use_random: bool = False
    rng: np.random.Generator = np.random.default_rng()
    use_adaptive: bool = False
    min_adaptive_relaxation_coefficient: float = 0.0
    max_adaptive_niter: int = 10000
    use_cache: bool = False
    cache_size: int = 10000
    cache: Optional[ClosestLRUCache] = None
    cache_keygen: Optional[Callable] = None

    _tau: npt.NDArray[np.float_] = field(init=False, repr=False)
    _Tex: npt.NDArray[np.float_] = field(init=False, repr=False)

    def __post_init__(self):
        if self.use_cache and self.cache_keygen is None:
            raise RuntimeError('cache_keygen must be provided if cache is used')

    @property
    def Tkin(self: NonLTESource) -> float:
        return self.mutable_params.Tkin

    @property
    def collision_density(self: NonLTESource) -> Dict[str, float]:
        return self.mutable_params.collision_density

    @property
    def background(self: NonLTESource) -> Union[Continuum, float]:
        return self.mutable_params.background

    @property
    def escape_probability(self: NonLTESource) -> EscapeProbability:
        return self.mutable_params.escape_probability

    @property
    def column(self: NonLTESource) -> float:
        return self.mutable_params.column

    @property
    def dV(self: NonLTESource) -> float:
        return self.mutable_params.dV

    @property
    def velocity(self: NonLTESource) -> float:
        return self.mutable_params.velocity

    @property
    def frequency(self: NonLTESource) -> npt.NDArray[np.float_]:
        return self.molecule.radiative_transitions.frequencies * (1.0 - self.velocity / ckm)

    @property
    def tau(self: NonLTESource) -> npt.NDArray[np.float_]:
        self.run()
        return self._tau

    @property
    def Tex(self: NonLTESource) -> npt.NDArray[np.float_]:
        self.run()
        return self._Tex

    def get_collisional_rate(self: NonLTESource) -> npt.NDArray[np.float_]:
        Tkin = self.mutable_params.Tkin
        partner_name_standardizer = self.molecule.partner_name_standardizer

        collision_density = dict()
        for partner_name, density in self.mutable_params.collision_density.items():
            collision_density[partner_name_standardizer[partner_name]] = density

        collisional_transitions = self.molecule.collisional_transitions
        if ('H2' in collision_density and 'H2' not in collisional_transitions
                and 'pH2' not in collision_density and 'pH2' in collisional_transitions
                and 'oH2' not in collision_density and 'oH2' in collisional_transitions):
            # calculate ortho-to-para ratio if H2 density is given but data file has o- and p-H2,
            # equation extracted from radex
            opr = min(3.0, 9.0 * math.exp(-170.6 / Tkin))
            ofrac = opr / (opr + 1.0)
            pfrac = 1.0 - ofrac
            H2_density = collision_density.pop('H2')
            collision_density['pH2'] = H2_density * pfrac
            collision_density['oH2'] = H2_density * ofrac

        return self._collisional_rate_helper(Tkin, frozenset(collision_density.items()))

    @lru_cache
    def _collisional_rate_helper(self: NonLTESource, Tkin: float, collision_density: FrozenSet[Tuple[str, float]]) -> npt.NDArray[np.float_]:
        levels = self.molecule.levels
        collisional_transitions = self.molecule.collisional_transitions

        crate: npt.NDArray[np.float_] = np.zeros(
            (levels.num_levels, levels.num_levels), dtype=float)

        # calculate rate coefficients multiplied by density.
        for partner_name, density in collision_density:
            num_transitions = collisional_transitions[partner_name].num_transitions
            iup = collisional_transitions[partner_name].upper_level_indices
            ilo = collisional_transitions[partner_name].lower_level_indices
            colld = self.molecule.collisional_rate_coefficients_getter[partner_name](
                Tkin)
            self._set_downward_rates(density, num_transitions, iup, ilo, colld, crate)

        # calculate upward rates from detail balance
        self._set_upward_rates_detail_balance(
            Tkin, levels.num_levels, levels.ediff, levels.gratio, crate)

        return crate

    @staticmethod
    @nb.jit
    def _set_downward_rates(density: float, num_transitions: int, iup: npt.NDArray[np.int_], ilo: npt.NDArray[np.int_], colld: npt.NDArray[np.float_], crate: npt.NDArray[np.float_]):
        for i in range(num_transitions):
            crate[iup[i], ilo[i]] += density * colld[i]

    @staticmethod
    @nb.jit
    def _set_upward_rates_detail_balance(Tkin: float, num_levels: int, ediff: npt.NDArray[np.float_], gratio: npt.NDArray[np.float_], crate: npt.NDArray[np.float_]):
        einv = -h * ccm / (k * Tkin)
        for iup in range(num_levels):
            for ilo in range(iup):
                crate[ilo, iup] = gratio[iup, ilo] * \
                    np.exp(einv * ediff[iup, ilo]) * crate[iup, ilo]

    def set_rate_matrix(self: NonLTESource, beta: npt.NDArray[np.float_], yrate: npt.NDArray[np.float_]):
        levels = self.molecule.levels
        radiative_transitions = self.molecule.radiative_transitions
        background = self.mutable_params.background

        num_transitions = radiative_transitions.num_transitions
        iup = radiative_transitions.upper_level_indices
        ilo = radiative_transitions.lower_level_indices

        Aul = radiative_transitions.spontaneous_decay_rates
        Bul_J = radiative_transitions.get_Bul_J(background)
        Blu_J = radiative_transitions.get_Blu_J(background)

        num_levels = levels.num_levels
        crate = self.get_collisional_rate()
        self._set_collisional_rates(num_levels, crate, yrate)
        self._add_radiative_rates(num_transitions, iup, ilo,
                                  Aul, Bul_J, Blu_J, beta, yrate)

    @staticmethod
    @nb.jit
    def _set_collisional_rates(num_levels: int, crate: npt.NDArray[np.float_], yrate: npt.NDArray[np.float_]):
        for i in range(num_levels):
            ctot = 0.0
            for j in range(num_levels):
                yrate[j, i] = -crate[i, j]
                ctot += crate[i, j]
            yrate[i, i] = ctot

    @staticmethod
    @nb.jit
    def _add_radiative_rates(num_transitions: int, iup: npt.NDArray[np.int_], ilo: npt.NDArray[np.int_], Aul: npt.NDArray[np.float_], Bul_J: npt.NDArray[np.float_], Blu_J: npt.NDArray[np.float_], beta: npt.NDArray[np.float_], yrate: npt.NDArray[np.float_]):
        for i in range(num_transitions):
            ul = (Aul[i] + Bul_J[i]) * beta[i]
            lu = Blu_J[i] * beta[i]
            yrate[iup[i], iup[i]] += ul
            yrate[iup[i], ilo[i]] -= lu
            yrate[ilo[i], iup[i]] -= ul
            yrate[ilo[i], ilo[i]] += lu

    def set_escape_probability(self: NonLTESource, tau: npt.NDArray[np.float_], beta: npt.NDArray[np.float_]):
        radiative_transitions = self.molecule.radiative_transitions
        num_transitions = radiative_transitions.num_transitions

        escape_probability = self.mutable_params.escape_probability
        escape_probability.set_probability(num_transitions, tau, beta)

    def set_optical_depth(self: NonLTESource, xpop: npt.NDArray[np.float_], tau: npt.NDArray[np.float_]):
        radiative_transitions = self.molecule.radiative_transitions

        num_transitions = radiative_transitions.num_transitions
        cddv = self.mutable_params.column / self.mutable_params.dV
        iup = radiative_transitions.upper_level_indices
        ilo = radiative_transitions.lower_level_indices
        Aul = radiative_transitions.spontaneous_decay_rates
        ediff = radiative_transitions.ediff
        gratio = radiative_transitions.gratio

        self._set_optical_depth_helper(
            num_transitions, cddv, iup, ilo, Aul, ediff, gratio, xpop, tau)

    @staticmethod
    @nb.jit
    def _set_optical_depth_helper(num_transitions: int, cddv: float, iup: npt.NDArray[np.int_], ilo: npt.NDArray[np.int_], Aul: npt.NDArray[np.float_], ediff: npt.NDArray[np.float_], gratio: npt.NDArray[np.float_], xpop: npt.NDArray[np.float_], tau: npt.NDArray[np.float_]):
        fgaus = np.sqrt(16 * np.pi**3 / np.log(2)) * 1e5
        for i in range(num_transitions):
            tau[i] = cddv * (xpop[ilo[i]] * gratio[i] - xpop[iup[i]]) * \
                Aul[i] / (fgaus * ediff[i]**3)

    def set_excitation_temperature(self: NonLTESource, xpop: npt.NDArray[np.float_], Tex_old: npt.NDArray[np.float_], Tex: npt.NDArray[np.float_]):
        radiative_transitions = self.molecule.radiative_transitions

        num_transitions = radiative_transitions.num_transitions
        iup = radiative_transitions.upper_level_indices
        ilo = radiative_transitions.lower_level_indices
        ediff = radiative_transitions.ediff
        gratio = radiative_transitions.gratio

        self._set_excitation_temperature_helper(
            num_transitions, iup, ilo, ediff, gratio, xpop, Tex_old, Tex)

    @staticmethod
    @nb.jit
    def _set_excitation_temperature_helper(num_transitions: int, iup: npt.NDArray[np.int_], ilo: npt.NDArray[np.int_], ediff: npt.NDArray[np.float_], gratio: npt.NDArray[np.float_], xpop: npt.NDArray[np.float_], Tex_old: npt.NDArray[np.float_], Tex: npt.NDArray[np.float_]):
        fk = h * ccm / k
        for i in range(num_transitions):
            if gratio[i] * xpop[ilo[i]] == xpop[iup[i]]:
                Tex[i] = Tex_old[i]
            else:
                Tex[i] = fk * ediff[i] / np.log(gratio[i] * xpop[ilo[i]] / xpop[iup[i]])

    def get_convergence(self: NonLTESource, tau: npt.NDArray[np.float_], Tex_old: npt.NDArray[np.float_], Tex: npt.NDArray[np.float_]) -> Tuple[bool, bool]:
        radiative_transitions = self.molecule.radiative_transitions
        num_transitions = radiative_transitions.num_transitions
        return self._get_convergence_helper(self.ccrit, num_transitions, tau, Tex_old, Tex)

    @staticmethod
    @nb.jit
    def _get_convergence_helper(ccrit: float, num_transitions: int, tau: npt.NDArray[np.float_], Tex_old: npt.NDArray[np.float_], Tex: npt.NDArray[np.float_]) -> Tuple[bool, bool]:
        nthick = 0
        tsum = 0.0
        for i in range(num_transitions):
            if np.isnan(tau[i]):
                return False, True
            if tau[i] > 0.01:
                nthick += 1
                tsum += abs((Tex[i] - Tex_old[i]) / Tex[i])
        converged = nthick == 0 or tsum / nthick < ccrit
        restart = False
        return converged, restart

    def relaxation(self: NonLTESource, coefficient: float, value: npt.NDArray[np.float_], value_old: npt.NDArray[np.float_]):
        if coefficient != 1.0:
            self._relaxation_helper(coefficient, value, value_old)

    @staticmethod
    @nb.jit
    def _relaxation_helper(coefficient: float, value: npt.NDArray[np.float_], value_old: npt.NDArray[np.float_]):
        c = 1.0 - coefficient
        for i in range(value.size):
            value[i] += c * (value_old[i] - value[i])

    def run(self: NonLTESource):
        if not self.mutable_params.is_mutated():
            return

        xpop0 = None
        if self.use_cache and self.cache_keygen is not None:
            key = self.cache_keygen(self.mutable_params)
            if self.cache is None:
                object.__setattr__(self, 'cache', ClosestLRUCache(
                    self.cache_size, key.size))
            else:
                xpop0 = self.cache.get(key)

        levels = self.molecule.levels
        radiative_transitions = self.molecule.radiative_transitions
        background = self.mutable_params.background

        num_levels = levels.num_levels
        num_transitions = radiative_transitions.num_transitions

        tau: npt.NDArray[np.float_]
        beta: npt.NDArray[np.float_]
        yrate: npt.NDArray[np.float_]
        rhs: npt.NDArray[np.float_]
        xpop: npt.NDArray[np.float_]
        xpop_old: npt.NDArray[np.float_]
        Tex: npt.NDArray[np.float_]
        Tex_old: npt.NDArray[np.float_]

        maxiter = self.maxiter
        niter = 0
        max_xpop_relaxation_coefficient = self.xpop_relaxation_coefficient
        while maxiter > 0:
            maxiter -= 1

            if niter == 0:
                tau = np.zeros(num_transitions, dtype=float)
                beta = np.ones(num_transitions, dtype=float)

                yrate = np.empty((num_levels, num_levels), dtype=float)
                rhs = np.zeros(num_levels)
                rhs[0] = 1.0

                if xpop0 is not None:
                    self.set_optical_depth(xpop0, tau)
                    self.set_escape_probability(tau, beta)
            else:
                self.set_optical_depth(xpop, tau)
                self.set_escape_probability(tau, beta)

            self.set_rate_matrix(beta, yrate)
            yrate[0] = 1.0

            if niter != 0:
                xpop_old = xpop
            xpop = np.linalg.solve(yrate, rhs)
            np.clip(xpop, 1e-30, None, out=xpop)
            xpop /= xpop.sum()

            if niter == 0:
                Tex = np.empty(num_transitions)
                if isinstance(background, Continuum):
                    Tex_old = background.Tbg(radiative_transitions.frequencies)
                else:
                    Tex_old = np.full(num_transitions, background, dtype=float)
            else:
                Tex_old, Tex = Tex, Tex_old
            self.set_excitation_temperature(xpop, Tex_old, Tex)
            self.set_optical_depth(xpop, tau)
            converged, restart = self.get_convergence(tau, Tex_old, Tex)

            if restart or (self.use_adaptive and niter >= self.max_adaptive_niter and max_xpop_relaxation_coefficient > self.min_adaptive_relaxation_coefficient):
                niter = 0
                if self.use_adaptive:
                    max_xpop_relaxation_coefficient = max(
                        self.min_adaptive_relaxation_coefficient, 0.5 * max_xpop_relaxation_coefficient)
                continue

            if niter != 0:
                self.relaxation(self.Tex_relaxation_coefficient, Tex, Tex_old)
                if self.use_random:
                    xpop_relaxation_coefficient = self.rng.uniform(
                        0.0, max_xpop_relaxation_coefficient)
                else:
                    xpop_relaxation_coefficient = max_xpop_relaxation_coefficient
                self.relaxation(xpop_relaxation_coefficient, xpop, xpop_old)

            if converged and niter >= self.miniter:
                break

            niter += 1

        object.__setattr__(self, '_tau', tau)
        object.__setattr__(self, '_Tex', Tex)

        if self.use_cache and self.cache is not None:
            self.cache.put(key, xpop)

        return niter


@dataclass
class NonLTESimulation:
    # List of NonLTESource objects associated with this simulation
    source: List[NonLTESource]
    size: float                                                     # source size

    continuum: Continuum = field(default_factory=Continuum)         # Continuum object
    # aperture size for spectrum extraction [arcsec]
    aperture: Optional[float] = None

    ll: List[float] = field(
        default_factory=lambda: [-np.Infinity])  # lower limits [MHz]
    ul: List[float] = field(
        default_factory=lambda: [np.Infinity])  # upper limits [MHz]
    # FWHMs to simulate +/- line center
    sim_width: float = 10.0
    # resolution if simulating line profiles [MHz]
    res: float = 0.01
    # units for the simulation; accepts 'K', 'mK', 'Jy/beam'
    units: str = 'K'
    # flag for line profile simulation to be done with observations
    use_obs: bool = False
    # Observation object associated with this simulation
    observation: Optional[Observation] = None

    # Spectrum object associated with this simulation
    spectrum: Spectrum = field(default_factory=Spectrum)
    beam_dilution: float = field(init=False)

    def __post_init__(self: NonLTESimulation):
        # cast to list if needed
        if not isinstance(self.source, Iterable):
            self.source = [self.source]

        if not isinstance(self.ll, Iterable):
            self.ll = [self.ll]

        if not isinstance(self.ul, Iterable):
            self.ul = [self.ul]

        # merge and sort ll ul intervals
        self.ll, self.ul = map(lambda x: list(x), zip(
            *_merge_intervals(zip(self.ll, self.ul))))

        # check if observation is provided
        if self.use_obs:
            if self.observation is None:
                raise RuntimeError('use_obs is True but observation is not provided')
        if self.units in ['Jy/beam', 'Jy']:
            if self.observation is None:
                raise RuntimeError(
                    f'units is {self.units} but observation is not provided')

        # finished setting default values, the rest is done in _update()
        self._update()

    def _get_tau_Iv(self: NonLTESimulation, source: NonLTESource, freq: npt.NDArray[np.float_], drawn_indices: npt.NDArray[np.int_]):
        tau = np.zeros_like(freq)
        Iv = np.zeros_like(freq)

        for freq_, Tex_, tau_ in zip(source.frequency[drawn_indices], source.Tex[drawn_indices], source.tau[drawn_indices]):
            dfreq = source.dV / ckm * freq_
            two_sigma_sq = dfreq**2 / (4 * np.log(2))
            lo = np.searchsorted(freq, freq_ - self.sim_width * dfreq, side='left')
            hi = np.searchsorted(freq, freq_ + self.sim_width * dfreq, side='right')
            f = freq[lo:hi]
            t = tau_ * np.exp(-(f - freq_)**2 / two_sigma_sq)
            tau[lo:hi] += t
            Iv[lo:hi] += _planck(Tex_, f) * t  # TODO: double-check this expression

        mask = tau > 0.0
        Iv[mask] *= -np.expm1(-tau[mask]) / tau[mask]
        return tau, Iv

    def _update(self: NonLTESimulation):
        # first, set up the frequency grid
        if self.use_obs:
            assert isinstance(self.observation, Observation)
            frequency = np.copy(self.observation.spectrum.frequency)
            dV_max = max(source.dV for source in self.source)
            combined_intervals = _merge_intervals(zip(
                frequency * (1.0 - self.sim_width * dV_max / ckm),
                frequency * (1.0 + self.sim_width * dV_max / ckm)
            ))
            drawn_intervals = [
                (left / (1.0 - self.sim_width * dV_max / ckm),
                 right / (1.0 + self.sim_width * dV_max / ckm))
                for left, right in combined_intervals
            ]
        else:
            combined_lines_intervals = _merge_intervals(
                [(float(freq * (1 - self.sim_width * source.dV / ckm)),
                  float(freq * (1 + self.sim_width * source.dV / ckm)))
                 for source in self.source for freq in source.frequency]
            )
            drawn_intervals = list(zip(self.ll, self.ul))
            frequency_generation_intervals = _intersect_intervals(
                combined_lines_intervals, drawn_intervals)
            if frequency_generation_intervals:
                frequency = np.concatenate([np.arange(s, e, self.res)
                                           for s, e in frequency_generation_intervals])
            else:
                frequency = np.empty(0)
        # lines to be drawn for each source
        drawn_indices_list = [_find_in_intervals(
            source.frequency, drawn_intervals) for source in self.source]

        # second, compute optical depth and intensity for each source, add to the total intensity
        intensity = np.zeros_like(frequency)
        cumulative_tau = np.zeros_like(frequency)
        for source, drawn_indices in zip(self.source, drawn_indices_list):
            tau, Iv = self._get_tau_Iv(source, frequency, drawn_indices)
            intensity += Iv * np.exp(-cumulative_tau)
            cumulative_tau += tau

        # third, compute the contribution from background continuum source, and subtract the continuum
        intensity += self.continuum.Ibg(frequency) * np.expm1(-cumulative_tau)

        # fourth, correct for beam dilution for gaussian beam or uniform aperture
        if self.observation is not None:
            if self.observation.observatory.sd is True:
                intensity, beam_dilution = _apply_beam(
                    frequency, intensity, self.size, self.observation.observatory.dish, return_beam=True)
            if self.observation.observatory.array is True and self.aperture is not None:
                intensity, beam_dilution = _apply_aperture(
                    intensity, self.size, self.aperture, return_beam=True)

        # fifth, set units
        if self.units in ['K', 'mK']:
            intensity = _rayleigh_jeans_temperature(intensity, frequency)
            if self.units == 'mK':
                intensity *= 1e3
        if self.units in ['Jy/beam', 'Jy']:
            assert isinstance(self.observation, Observation)
            omega = self.observation.observatory.synth_beam[0] * \
                self.observation.observatory.synth_beam[1]
            sr_per_beam = omega * np.pi / (206265**2 * 4 * np.log(2))
            intensity *= sr_per_beam

        self.spectrum.freq_profile = frequency
        self.spectrum.int_profile = intensity
        self.beam_dilution = beam_dilution
