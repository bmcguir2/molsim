
from typing import Tuple, Union, Type, NamedTuple
from collections import namedtuple
from abc import ABC, abstractmethod
import sys

import numpy as np


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
        self._param = UniformParameter("test", 0., -1., 1.)

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
    def ln_likelihood(self, value: float):
        raise NotImplementedError


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
            return 0.
        else:
            return -np.inf
    
    def __repr__(self):
        return super().__repr__()


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
            lnlikelihood = np.log(1. / (np.sqrt(2. * np.pi) * var))
            lnlikelihood -= 0.5 * ((value - mu)**2 / (var**2))
            return lnlikelihood
        else:
            return -np.inf


class BaseModel:
    def __init__(self, model_func, *distributions):
        super().__init__()
        self._distributions = list(distributions)
        self.model_func = model_func

    def prior_log_likelihood(self, *values) -> float:
        assert len(values) == len(self._distributions)
        for index, (dist, value) in enumerate(zip(self._distributions, values)):
            if index == 0:
                lnlikelihood = dist.ln_likelihood(value)
            else:
                lnlikelihood += dist.ln_likelihood(value)
        return lnlikelihood
            
    def log_likelihood(self, target, target_sigma, model) -> float:
        inv_sigmasq = 1.0 / (target_sigma ** 2)
        tot_lnlike = np.sum(
            (target - model) ** 2 * inv_sigmasq - np.log(inv_sigmasq)
        )
        return -0.5 * tot_lnlike

    def total_likelihood(self, target, target_sigma, *values):
        prior = self.prior_log_likelihood(*values)
        model = self.model_func(*values)


class BaseHelper(ABC):
    @property
    @abstractmethod
    def target(self):
        # this stores the spectral data we are trying
        # to model/condition with
        return self._target

    @property
    @abstractmethod
    def molecule(self):
        # this abstract property stores the catalog details
        return self._molecule

    @property
    @abstractmethod
    def model(self):
        return self._model
    
    @property
    @abstractmethod
    def traces(self):
        return self._traces
    
    @abstractmethod
    def build_model(self):
        raise NotImplementedError
    
    def posterior_spectrum(self):
        """
        This method is supposed to build a spectrum using the posterior predictive.
        """
        raise NotImplementedError

    def __call__(self, *args):
        return self.sample(*args)
    
    def reset(self):
        self._traces = None

    def __post_init__(self):
        self.reset()
