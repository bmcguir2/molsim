from typing import Tuple, Union, Type, NamedTuple, Callable, List
from collections import namedtuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
import sys

import numpy as np
import emcee

from molsim.mcmc import compute


UniformParameter = namedtuple("UniformParameter", "min max")
GaussianParameter = namedtuple("GaussianParameter", "mu var min max")


def str2class(classname: str):
    obj = getattr(sys.modules[__name__], classname, None)
    if not obj:
        raise NameError("Invalid name for class!")
    else:
        return obj


class AbstractDistribution(ABC):
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._param = UniformParameter(-1.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.name()}, limits {self.limits()}"

    @property
    @abstractmethod
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def param(self) -> NamedTuple:
        return self._param

    @abstractmethod
    def ln_likelihood(self, value: float) -> float:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, **kwargs):
        raise NotImplementedError

    def __call__(self, value: float) -> float:
        return self.ln_likelihood(value)

    def initial_value(self) -> float:
        param = self.param()
        return (param.min + param.max) / 2.0


class UniformLikelihood(AbstractDistribution):
    def __init__(self, name: str, param: UniformParameter):
        super().__init__(name)
        self._param = param

    @property
    def name(self) -> str:
        return self._name

    @property
    def param(self) -> UniformParameter:
        return self._param

    @classmethod
    def from_dict(cls, name: str, **kwargs):
        params = UniformParameter(**kwargs)
        return cls(name, params)

    @classmethod
    def from_values(cls, name: str, min=-np.inf, max=np.inf):
        param = UniformParameter(min, max)
        return cls(name, param)

    def ln_likelihood(self, value: float) -> float:
        """
        Return the log likelihood based on an uninformative prior;
        if the value is within the constraints, we return a likelihood
        of one and outside of the constraints, we make it infinitely
        unlikely.

        Parameters
        ----------
        value : float
            [description]

        Returns
        -------
        float
            [description]
        """
        if self._param.min <= value <= self._param.max:
            return 0.0
        else:
            return -np.inf


class GaussianLikelihood(AbstractDistribution):
    def __init__(self, name: str, param: GaussianParameter):
        """
        [summary]

        Parameters
        ----------
        name : str
            [description]
        limits : Tuple[float]
            Two-tuple containing the Gaussian center (mu) and variance (var)
            that defines the normal distribution N~(mu, var)
        """
        super().__init__(name)
        self._param = param

    @property
    def name(self) -> str:
        return self._name

    @property
    def param(self) -> GaussianParameter:
        return self._param

    def ln_likelihood(self, value: float) -> float:
        if self._param.min <= value <= self._param.max:
            mu, var = self._param.mu, self._param.var
            lnlikelihood = (
                np.log(1.0 / (np.sqrt(2 * np.pi) * var))
                - 0.5 * (value - mu) ** 2 / var ** 2
            )
            return lnlikelihood
        else:
            return -np.inf

    @classmethod
    def from_npy_chain(cls, name: str, chain: np.ndarray, min=0.0, max=np.inf):
        percentiles = np.percentile(chain, [16.0, 50.0, 84.0])
        var = np.mean([percentiles[0], percentiles[-1]])
        mu = percentiles[1]
        param = GaussianParameter(mu, var, min, max)
        return cls(name, param)

    @classmethod
    def from_values(cls, name: str, mu: float, var: float, min=0.0, max=np.inf):
        param = GaussianParameter(mu, var, min, max)
        return cls(name, param)

    @classmethod
    def from_dict(cls, name: str, **kwargs):
        params = GaussianParameter(**kwargs)
        return cls(name, params)


class AbstractModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_log_likelihood(self, parameters: np.ndarray) -> float:
        raise NotImplementedError


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
        simulation = self.simulate_spectrum(parameters)
        inv_sigmasq = 1.0 / (self.spectrum.noise ** 2.0)
        tot_lnlike = np.sum(
            (self.spectrum.intensity - simulation) ** 2 * inv_sigmasq
            - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike


class EmceeHelper(object):
    def __init__(self, num_parameters: int, initial: np.ndarray):
        super().__init__()
        self.ndim = num_parameters
        self.chain = None
        self.positions = None

    def sample(
        self,
        model: AbstractModel,
        walkers: int = 100,
        iterations: int = 1000,
        workers: int = 4,
        scale: float = 1e-3,
    ):
        positions = np.tile(self.initial, (walkers, 1))
        scrambler = np.ones_like(positions)
        scrambler += np.random.uniform(-scale, scale, (walkers, self.ndim))
        positions *= scrambler
        # run the MCMC sampling
        with Pool(workers) as pool:
            sampler = emcee.EnsembleSampler(
                walkers,
                self.ndim,
                compute_model_likelihoods,
                args=tuple(model),
                pool=pool,
            )
            sampler.run_mcmc(positions, iterations, progress=True)
        self.chain = sampler.chain
        self.positions = sampler.get_last_sample()


def compute_model_likelihoods(parameters: np.ndarray, model: AbstractModel) -> float:
    prior = model.compute_prior_likelihood(parameters)
    if np.isfinite(prior):
        return prior + model.compute_loglikelihood(parameters)
    else:
        return -np.inf

