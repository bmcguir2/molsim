from typing import Type, Dict, Union, Callable, List
from pathlib import Path
from dataclasses import dataclass

import pymc3 as pm
import numpy as np
import numexpr as ne
from loguru import logger

from molsim.mcmc.base import AbstractModel, AbstractDistribution
from molsim.mcmc import compute
from molsim.utils import load_yaml, find_limits
from molsim.classes import Source, Molecule, Simulation, Observation
from molsim.functions import sum_spectra


@dataclass
class SingleComponent(AbstractModel):
    source_size: AbstractDistribution
    vlsr: AbstractDistribution
    Ncol: AbstractDistribution
    Tex: AbstractDistribution
    dV: AbstractDistribution
    observation: Observation

    def __post_init__(self):
        self._distributions = [
            self.source_size,
            self.vlsr,
            self.Ncol,
            self.Tex,
            self.dV,
        ]

    def __len__(self) -> int:
        return len(self._distributions)

    def _get_components(self):
        return self._distributions

    def initialize_values(self):
        initial = [param.initial_value() for param in self._distributions]
        return initial

    def simulate_spectrum(
        self,
        parameters: np.ndarray,
        molecule: Molecule,
    ) -> np.ndarray:
        size, vlsr, ncol, Tex, dV = parameters
        source = Source("", vlsr, size, column=ncol, Tex=Tex, dV=dV)
        min_freq, max_freq = find_limits(observation.spectrum.frequency)
        simulation = Simulation(
            mol=Molecule,
            ll=min_freq,
            ul=max_freq,
            observation=self.observation,
            source=source,
            line_profile="gaussian",
            use_obs=True
        )
        return simulation

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        lnlikelihood = sum(
            [
                dist.ln_likelihood(value)
                for dist, value in zip(self._distributions, parameters)
            ]
        )
        return lnlikelihood

    def compute_log_likelihood(self, parameters: np.ndarray) -> float:
        """
        Calculate the negative log likelihood, given a set of parameters
        and our observed data.

        Parameters
        ----------
        parameters : np.ndarray
            [description]

        Returns
        -------
        float
            [description]
        """
        simulation = self.simulate_spectrum(parameters)
        inv_sigmasq = 1.0 / (self.spectrum.noise ** 2.0)
        tot_lnlike = np.sum(
            (self.spectrum.intensity - simulation.spectrum.int_profile) ** 2 * inv_sigmasq
            - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike


class MultiComponent(SingleComponent):
    """
    Implementation of a multi component model. This type of model extends
    the parameters expected with parameters for each component like so:
    
    [source_size1, source_size2, vlsr1, vlsr2,...Tex, dV]
    
    So that there is an arbitrary number of components, providing each
    component has a source size, radial velocity, and column density.
    """

    def __init__(
        self,
        source_sizes: List[AbstractDistribution],
        vlsrs: List[AbstractDistribution],
        Ncols: List[AbstractDistribution],
        Tex: AbstractDistribution,
        dV: AbstractDistribution,
        observation: Observation,
    ):
        super().__init__(source_sizes, vlsrs, Ncols, Tex, dV, observation)
        assert len(source_sizes) == len(vlsrs) == len(Ncols)
        self.components = list()
        # these are not used in preference of `self.components` instead
        self.source_size = self.vlsr = self.Ncol = self._distributions = None
        for ss, vlsr, Ncol in zip(source_sizes, vlsrs, Ncols):
            self.components.append(
                SingleComponent(
                    ss,
                    vlsr,
                    Ncol,
                    Tex,
                    dV,
                    observation,
                )
            )

    def __len__(self) -> int:
        return len(self.components) * 3 + 2

    def _get_components(self):
        params = list()
        for index, component in enumerate(self.components):
            if index != len(self.components):
                params.extend(component._distributions[:3])
            else:
                params.extend(component._distributions)
        return params

    def initialize_values(self):
        params = list()
        for index, component in enumerate(self.components):
            values = component.initialize_values()[:-2]
            if index != len(self.components):
                values = values[:-2]
            params.extend(values)
        return params

    def _get_component_parameters(
        self, parameters: np.ndarray, component: int
    ) -> np.ndarray:
        subparams = parameters[component:-2:4]
        # tack on the excitation temperature and linewidth, which are global
        subparams = np.append(subparams, [parameters[-2], parameters[-1]])
        return subparams

    def simulate_spectrum(
        self,
        parameters: np.ndarray,
        molecule: Molecule,
    ) -> np.ndarray:
        simulations = list()
        for index, component in enumerate(self.components):
            # take every third element, corresponding to a parameter for
            # a particular component. Skip the last two, which are
            subparams = self._get_component_parameters(parameters, index)
            simulations.append(
                component.simulate_spectrum(subparams, molecule)
            )
        full_sim = sum_spectra(simulations)
        return full_sim

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        for index, component in enumerate(self.components):
            subparams = self._get_component_parameters(parameters, index)
            if index == 0:
                lnlikelihood = component.compute_prior_likelihood(subparams)
            else:
                lnlikelihood += component.compute_prior_likelihood(subparams)
        return lnlikelihood
