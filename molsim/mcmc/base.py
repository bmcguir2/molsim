
from typing import Tuple, Union, Type

from abc import ABC, abstractmethod
import pymc3 as pm


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

    def sample(self, nsamples: int, tune: int, n_workers: int = 4, **kwargs):
        with self._model:
            self._traces = pm.sample(nsamples, tune=tune, cores=n_workers, **kwargs)
        return self._traces
    
    def find_MAP_estimate(self):
        return pm.find_MAP(model=self._model)
    
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
