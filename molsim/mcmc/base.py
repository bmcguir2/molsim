from typing import Tuple, Union, Type, NamedTuple, Callable, List
from collections import namedtuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
import sys
import json

import numpy as np
import emcee
import arviz
from loguru import logger

# this makes sure that the full range of parameters are shown
# for pair plots
arviz.rcParams["plot.max_subplots"] = 120

from molsim.mcmc import compute

"""
These namedtuples are used to compactly represent parameters
for a likelihood function. Currently there are two implemented
based on what was used for the GOTHAM data: uniform and gaussian.
"""

UniformParameter = namedtuple("UniformParameter", "min max", defaults=(-np.inf, np.inf))
GaussianParameter = namedtuple(
    "GaussianParameter", "mu var min max", defaults=(0.0, 1.0, 0.0, np.inf)
)


class AbstractDistribution(ABC):
    """
    Base class for the likelihood distributions.

    Defines the abstract methods and properties that are shared
    between the different types of distributions; for example,
    all distributions need a way to calculate the log likelihood.

    New distributions, beyond uniform and Gaussians, should inherit
    from this class.
    """

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._param = UniformParameter()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.name}, {self.param}"

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

    @abstractmethod
    def initial_value(self) -> float:
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

    @classmethod
    def from_dict(cls, name: str, **kwargs):
        params = UniformParameter(**kwargs)
        return cls(name, params)

    @classmethod
    def from_values(cls, name: str, min=-np.inf, max=np.inf):
        param = UniformParameter(min, max)
        return cls(name, param)

    def initial_value(self) -> float:
        return (self.param.min + self.param.max) / 2.0

    def ln_likelihood(self, value: float) -> float:
        """
        Return the log likelihood based on an uninformative prior;
        if the value is within the constraints, we return a likelihood
        of one and outside of the constraints, we make it infinitely
        unlikely.

        Parameters
        ----------
        value : float
            Value of the parameter

        Returns
        -------
        float
            0. if between the limits, negative infinity otherwise
        """
        if self.param.min <= value <= self.param.max:
            return 0.0
        else:
            return -np.inf


class GaussianLikelihood(AbstractDistribution):
    def __init__(self, name: str, param: GaussianParameter):
        """
        A likelihood class representing a normal distribution, centered

        Parameters
        ----------
        name : str
            [description]
        param : GaussianParameter
            Requires a `GaussianParameter` namedtuple, with center mu
            and variance var, and 
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
        if self.param.min <= value <= self.param.max:
            mu, var = self.param.mu, self.param.var
            lnlikelihood = (
                np.log(1.0 / (np.sqrt(2 * np.pi) * var))
                - 0.5 * (value - mu) ** 2 / var ** 2
            )
            return lnlikelihood
        else:
            return -np.inf

    def initial_value(self) -> float:
        # for a Gaussian parameter, return the average value
        return self.param.mu

    @classmethod
    def from_npy_chain(cls, name: str, chain: np.ndarray, min=0.0, max=np.inf):
        """
        Uses a chain of parameter samples, as a NumPy 1D array, to compute the mean
        and variance as parameters for a new `GaussianLikelihood` instance.

        Parameters
        ----------
        name : str
            [description]
        chain : np.ndarray
            [description]
        min, max : float, optional
            Min/max values that this parameter can take, by default 0.0 and np.inf.
            If the sampled value exceeds this, we return negative infinity as the
            likelihood.
        """
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
    @abstractmethod
    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_log_likelihood(self, parameters: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def prior_constraint(self, parameters: np.ndarray):
        pass


class EmceeHelper(object):
    def __init__(self, initial: np.ndarray):
        super().__init__()
        self.initial = initial
        self.ndim = len(initial)
        self.chain = None
        self._positions = None
        self.sampler = None
        logger.info("MCMC analysis using emcee and Molsim")
        logger.info(
            f"NumPy version: {np.__version__}, Emcee version: {emcee.__version__}"
        )
        logger.info(f"Initial parameters: {initial}")

    @property
    def posterior(self):
        """
        To extract a number of burn-ins, you need to index using `xarray`
        syntax. In this case, it'll look like:
        
        ```helper.posterior.sel(draw=slice(3000,None))```
        
        To get draws 3000 to the end. Unfortunately, -1000 does not appear
        to work for this syntax.

        Returns
        -------
        [type]
            [description]
        """
        return arviz.convert_to_inference_data(self.chain)

    @staticmethod
    def likelihood_checks(model: AbstractModel, parameters: np.ndarray):
        logger.info(f"Performing prior log likelihood check.")
        prior = model.compute_prior_likelihood(parameters)
        if not np.isfinite(prior):
            raise ValueError(
                f"Prior likelihood for initial parameters is {prior}! Check your values."
            )
        logger.info(f"Passed—{prior:.4f}")
        logger.info(f"Performing log likelihood check.")
        ln = model.compute_log_likelihood(parameters)
        if not np.isfinite(ln):
            raise ValueError(
                f"Log likelihood for initial parameters is {ln}! Check your data."
            )
        logger.info(f"Passed—{ln:.4f}")

    def summary(self, model: AbstractModel) -> "DataFrame":
        """
        Generate a summary table of the posterior, using a model to get the names
        of the parameters.

        Parameters
        ----------
        model : AbstractModel
            [description]

        Returns
        -------
        [type]
            [description]
        """
        param_names = model.get_names()
        summary = arviz.summary(self.posterior)
        summary.index = param_names
        return summary

    def sample(
        self,
        model: AbstractModel,
        walkers: int = 100,
        iterations: int = 1000,
        workers: int = 1,
        scale: float = 1e-2,
    ):
        logger.info(f"Performing sampling with model:")
        logger.info(f"{model}")
        self.likelihood_checks(model, self.initial)
        # set up walker positions, and move them by a small percentage
        positions = np.tile(self.initial, (walkers, 1))
        scrambler = np.ones_like(positions)
        scrambler += np.random.uniform(-scale, scale, (walkers, self.ndim))
        positions *= scrambler
        # run the MCMC sampling
        if workers > 1:
            logger.info(f"Using multiprocessing for sampling with {workers} processes.")
            with Pool(workers) as pool:
                sampler = emcee.EnsembleSampler(
                    walkers,
                    self.ndim,
                    compute_model_likelihoods,
                    args=[model,],
                    pool=pool,
                )
                try:
                    sampler.run_mcmc(positions, iterations, progress=True)
                except ValueError as error:
                    logger.info(f"Sampling broke during evaluation of likelihood.")
                    logger.info(f"Dumping sampler positions to dump.npz")
                    positions = sampler.get_last_sample().coords
                    np.save("dump.npz", positions)
                    logger.error(error)
        else:
            logger.info(f"Using single process for sampling.")
            sampler = emcee.EnsembleSampler(
                walkers, self.ndim, compute_model_likelihoods, args=[model,],
            )
            try:
                sampler.run_mcmc(positions, iterations, progress=True)
            except ValueError as error:
                logger.info(f"Sampling broke during evaluation of likelihood.")
                logger.info(f"Dumping sampler positions to dump.npz")
                positions = sampler.get_last_sample().coords
                np.save("dump.npz", positions)
                logger.error(error)
        self.sampler = sampler
        self.chain = sampler.chain
        self.positions = sampler.get_last_sample()
        report = arviz.summary(self.posterior.posterior.isel(draw=slice(-30, None)))
        logger.info("Summary of last 30 steps of sampling:")
        logger.info(report)

    def save_posterior(self, filename: str) -> None:
        posterior = self.posterior
        arviz.to_netcdf(posterior, filename)
        logger.info(f"Saved posterior samples to {filename}.")

    @classmethod
    def from_netcdf(cls, netcdf_path: str, restart: bool = False):
        logger.info(f"Loading NetCDF chain; restarti = {restart}")
        samples = arviz.from_netcdf(netcdf_path)
        # if we're restarting sampling, take the last position
        if restart:
            last = samples.posterior.isel(draw=-1).mean(dim=["chain"]).to_array()
            initial = np.array(last)[0]
        # generate the initial values from the mean of the posterior
        else:
            initial = np.array(
                samples.posterior.mean(dim=["chain", "draw"]).to_array()
            )[0]
        helper_obj = cls(initial)
        helper_obj.chain = np.array(samples.posterior.to_array()).squeeze()
        return helper_obj

    @staticmethod
    def chains_to_prior(chains: np.ndarray, distributions: List[NamedTuple]):
        # make sure the number of parameters in the chain matches
        # the number of input distributions
        assert len(distributions) == chains.shape[-1]
        for index, dist in enumerate(distributions):
            dist_type = dist.__name__
            if dist_type == "UniformParameter":
                pass
            elif dist_type == "GaussianParameter":
                pass
            else:
                raise NotImplementedError(f"Unrecognized parameter type! {dist_type}")

    def sample_posterior(
        self, nsamples: int, nparams: int = 14, rng: np.random.Generator = None
    ) -> np.ndarray:
        """
        Take a random sample from the posterior. This is useful for simulating
        spectra for the purpose of illustrating how uncertainty in the model
        parameters is reflected in the result.

        Parameters
        ----------
        nsamples : int
            Number of random samples to draw from the posterior.
        nparams : int, optional
            Dimensionality of the model, i.e. number of parameters.
            By default 14
        rng : np.random.Generator, optional
            Instance of a NumPy RNG, by default None, which
            creates one.

        Returns
        -------
        np.ndarray
            Random samples drawn from the posterior.
        """
        samples = (
            np.array(self.posterior.posterior.to_array()).squeeze().reshape(-1, nparams)
        )
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(samples, nsamples, axis=0)

    def posterior_to_json(
        self, name: str, model: AbstractModel, return_dict: bool = False
    ) -> Union[dict, None]:
        summary = self.summary(model)
        n_components = len(model.components)
        output = dict()
        for parameter in ["SourceSize", "VLSR", "NCol"]:
            for component in range(n_components):
                key = f"{parameter}_{component}"
                # if the parameter hasn't been done yet, initialize the dictionary
                if parameter not in output:
                    output[parameter] = {"mean": [], "sd": []}
                # use pandas.loc to grab the value
                output[parameter]["mean"].append(summary.loc[key, "mean"])
                output[parameter]["sd"].append(summary.loc[key, "sd"])
        # loop over Tex and dV separately because we repeat them
        for parameter in ["Tex", "dV"]:
            output[parameter] = {
                "mean": [summary.loc[parameter, "mean"],] * n_components,
                "sd": [summary.loc[parameter, "sd"],] * n_components,
            }

        with open(f"{name}_mcmc_result.json", "w+") as write_file:
            json.dump(output, write_file, indent=4)
        if return_dict:
            return output
        else:
            return None


def compute_model_likelihoods(parameters: np.ndarray, model: AbstractModel) -> float:
    """
    Wrapper function used in `emcee` sampling calls. This implements the
    abstraction needed to bridge the two aspects of our code: parameters
    that are sampled, and a "model" that dictates the prior likelihoods
    and how a spectrum is simulated.

    Parameters
    ----------
    parameters : np.ndarray
        [description]
    model : AbstractModel
        [description]

    Returns
    -------
    float
        [description]
    """
    prior = model.compute_prior_likelihood(parameters)
    # if we're out of bounds for the prior, don't bother
    # performing the simulation
    if not np.isfinite(prior):
        return -np.inf
    ln = prior + model.compute_log_likelihood(parameters)
    return ln
