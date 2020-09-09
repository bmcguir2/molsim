from typing import Type, Dict, Union, Callable, List
from pathlib import Path
from dataclasses import dataclass

import pymc3 as pm
import numpy as np
import numexpr as ne
from loguru import logger
from joblib import load

from molsim.mcmc.base import AbstractModel, AbstractDistribution, UniformLikelihood, GaussianLikelihood
from molsim.mcmc import compute
from molsim.utils import load_yaml, find_limits
from molsim.classes import Source, Molecule, Simulation, Observation, Spectrum
from molsim.functions import sum_spectra
from molsim.file_handling import load_mol


@dataclass
class SingleComponent(AbstractModel):
    source_size: AbstractDistribution
    vlsr: AbstractDistribution
    Ncol: AbstractDistribution
    Tex: AbstractDistribution
    dV: AbstractDistribution
    observation: Observation
    molecule: Molecule

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
    ) -> np.ndarray:
        size, vlsr, ncol, Tex, dV = parameters
        source = Source("", vlsr, size, column=ncol, Tex=Tex, dV=dV)
        min_freq, max_freq = find_limits(self.observation.spectrum.frequency)
        simulation = Simulation(
            mol=self.molecule,
            ll=min_freq,
            ul=max_freq,
            observation=self.observation,
            source=source,
            line_profile="gaussian",
            use_obs=True
        )
        return simulation.spectrum.int_profile

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
        obs = self.observation.spectrum
        simulation = self.simulate_spectrum(parameters)
        inv_sigmasq = 1.0 / (obs.noise ** 2.0)
        tot_lnlike = np.sum(
            (obs.Tb - simulation) ** 2 * inv_sigmasq
            - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike

    @classmethod
    def from_yml(cls, yml_path: str):
        input_dict = load_yaml(yml_path)
        cls_dict = dict()
        # the two stragglers
        for key in input_dict.keys():
            if key != "observation":
                if hasattr(input_dict[key], "mu"):
                    dist = GaussianLikelihood
                else:
                    dist = UniformLikelihood
                cls_dict[key] = dist.from_values(**input_dict[key])
            else:
                # load in the observed data
                cls_dict["observation"] = load(input_dict["observation"])
        return cls(**cls_dict)


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
        molecule: Molecule
    ):
        super().__init__(source_sizes, vlsrs, Ncols, Tex, dV, observation, molecule)
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
                    molecule
                )
            )

    @classmethod
    def from_yml(cls, yml_path: str):
        input_dict = load_yaml(yml_path)
        cls_dict = dict()
        source_sizes, vlsrs, Ncols = list(), list(), list()
        # make sure the number of components is the same
        assert len(input_dict["source_sizes"]) == len(input_dict["vlsrs"]) == len(input_dict["Ncols"])
        n_components = len(input_dict["source_sizes"])
        # parse in all the different parameters
        for param_list, parameter in zip([source_sizes, vlsrs, Ncols], ["source_sizes", "vlsrs", "Ncols"]):
            for index in range(n_components):
                size_params = input_dict[parameter][index]
                size_params["name"] = f"{parameter}_{index}"
                if "mu" in size_params:
                    dist = GaussianLikelihood
                else:
                    dist = UniformLikelihood
                param_list.append(
                    dist.from_values(**size_params)
                )
            cls_dict[parameter] = param_list
        # the two stragglers
        for key in ["Tex", "dV"]:
            input_dict[key]["name"] = key
            if "mu" in input_dict[key]:
                dist = GaussianLikelihood
            else:
                dist = UniformLikelihood
            cls_dict[key] = dist.from_values(**input_dict[key])
        # load in the observed data
        cls_dict["observation"] = load(input_dict["observation"])
        cls_dict["molecule"] = load(input_dict["molecule"])
        return cls(**cls_dict)

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
        source_sizes = list()
        vlsrs = list()
        ncols = list()
        for index, component in enumerate(self.components):
            values = component.initialize_values()
            source_sizes.append(values[0])
            vlsrs.append(values[1])
            ncols.append(values[2])
            if index != len(self.components):
                final = values[-2:]
        params = list()
        for array in [source_sizes, vlsrs, ncols, final]:
            params.extend(array)
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
    ) -> np.ndarray:
        combined_intensity = np.zeros_like(self.observation.spectrum.frequency)
        for index, component in enumerate(self.components):
            # take every third element, corresponding to a parameter for
            # a particular component. Skip the last two, which are
            subparams = self._get_component_parameters(parameters, index)
            combined_intensity += component.simulate_spectrum(subparams)
        return combined_intensity

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        for index, component in enumerate(self.components):
            subparams = self._get_component_parameters(parameters, index)
            if index == 0:
                lnlikelihood = component.compute_prior_likelihood(subparams)
            else:
                lnlikelihood += component.compute_prior_likelihood(subparams)
        return lnlikelihood
