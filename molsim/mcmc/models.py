from typing import Type, Dict, Union, Callable
from pathlib import Path

import pymc3 as pm
import numpy as np
import numexpr as ne
from loguru import logger

from molsim.mcmc.base import AbstractModel, AbstractDistribution
from molsim.mcmc import compute
from molsim.utils import load_yaml


@dataclass
class SingleComponent(AbstractModel):
    source_size: AbstractDistribution
    vlsr: AbstractDistribution
    Ncol: AbstractDistribution
    Tex: AbstractDistribution
    dV: AbstractDistribution
    calc_Q: Union[Callable, str]
    spectrum: Type["DataChunk"]
    catalog: Type["Catalog"]
    dish_size: float = 100.0
    background_temperature: float = 2.725

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

    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        _, intensity = compute.build_synthetic_spectrum(
            *parameters,
            calc_Q=self.calc_Q,
            spectrum=self.spectrum,
            catalog=self.catalog,
            dish_size=self.dish_size,
            background_temperature=self.background_temperature,
        )
        return intensity

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        lnlikelihood = sum(
            [
                dist.ln_likelihood(value)
                for dist, value in zip(self._distributions, parameters)
            ]
        )
        return lnlikelihood

    def compute_log_likelihood(self, parameters: np.ndarray) -> float:
        simulation = self.simulate_spectrum(parameters)
        inv_sigmasq = 1.0 / (self.spectrum.noise ** 2.0)
        tot_lnlike = np.sum(
            (self.spectrum.intensity - simulation) ** 2 * inv_sigmasq
            - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike


class MultiComponent(AbstractModel):
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
        calc_Q: Union[Callable, str],
        spectrum: Type["DataChunk"],
        catalog: Type["Catalog"],
        dish_size: float = 100.0,
        background_temperature: float = 2.725,
    ):
        super().__init__()
        assert len(source_sizes) == len(vlsrs) == len(Ncols)
        self.components = list()
        for ss, vlsr, Ncol in zip(source_sizes, vlsrs, Ncols):
            self.components.append(
                SingleComponent(
                    ss,
                    vlsr,
                    Ncol,
                    Tex,
                    dV,
                    calc_Q,
                    spectrum,
                    catalog,
                    dish_size,
                    background_temperature,
                )
            )
        if type(calc_Q) == str:
            self.calc_Q = ne.NumExpr(calc_Q)
        elif callable(calc_Q):
            self.calc_Q = calc_Q
        else:
            raise NotImplementedError("Provided `calc_Q` is not a callable function or a string!")

    def __len__(self) -> int:
        return len(self.components) * 3 + 2

    @classmethod
    def from_yml(cls, yml_path: str):
        """
        The dictionary should be structured as such:
        
        

        Parameters
        ----------
        yml_path : str
            [description]
        """
        params = load_yaml(yml_path)
        

    def _get_components(self):
        params = list()
        for index, component in enumerate(self.components):
            if index != len(self.components):
                params.extend(component._distributions[:3])
            else:
                params.extend(component._distributions)
        return params

    def _get_component_parameters(
        self, parameters: np.ndarray, component: int
    ) -> np.ndarray:
        subparams = parameters[component:-2:4]
        # tack on the excitation temperature and linewidth, which are global
        subparams = np.append(subparams, [parameters[-2], parameters[-1]])
        return subparams

    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        Tex, dV = parameters[-2], parameters[-1]
        intensity = np.zeros_like(self.spectrum.frequency)
        for index, component in enumerate(self.components):
            # take every third element, corresponding to a parameter for
            # a particular component. Skip the last two, which are
            subparams = self._get_component_parameters(parameters, index)
            intensity += component.simulate_spectrum(
                subparams,
                calc_Q=self.calc_Q,
                spectrum=self.spectrum,
                catalog=self.catalog,
                dish_size=self.dish_size,
                background_temperature=self.background_temperature,
            )
        return intensity

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        for index, component in enumerate(self.components):
            subparams = self._get_component_parameters(parameters, index)
            if index == 0:
                lnlikelihood = component.compute_prior_likelihood(subparams)
            else:
                lnlikelihood += component.compute_prior_likelihood(subparams)
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
            (self.spectrum.intensity - simulation) ** 2 * inv_sigmasq
            - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike
