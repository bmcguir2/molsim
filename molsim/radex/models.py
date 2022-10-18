from dataclasses import dataclass
from typing import List, Union, Tuple
import numpy as np
from loguru import logger
from joblib import load
from scipy.optimize import minimize

from .classes import NonLTESource, MaserSimulation
from ..classes import Observation, Continuum
from ..mcmc.base import AbstractModel, AbstractDistribution, UniformLikelihood, GaussianLikelihood
from ..utils import load_yaml

@dataclass
class MultiComponentMaserModel(AbstractModel):
    Tbg: AbstractDistribution
    Tks: List[AbstractDistribution]
    nH2s: List[AbstractDistribution]
    vlsrs: List[AbstractDistribution]
    dVs: List[AbstractDistribution]
    Ncols: List[AbstractDistribution]
    source_sizes: List[float]
    continuum: Continuum
    collision_file: str
    radex_output: str
    observation: Observation
    aperture: float

    def __post_init__(self):
        # make sure all lists have the same length
        length = set(map(len, [self.Tks, self.nH2s, self.vlsrs, self.dVs, self.Ncols, self.source_sizes]))
        self.ncomponent = length.pop()
        if length:
            raise ValueError("Distribution lists have inconsistent length.")

        self._distributions = (
            [self.Tbg] + self.Tks + self.nH2s + self.vlsrs + self.dVs + self.Ncols
        )

        self.ll = self.observation.spectrum.frequency.min()
        self.ul = self.observation.spectrum.frequency.max()

    def __len__(self) -> int:
        return len(self._distributions)

    def _get_components(self):
        return self._distributions

    def get_names(self) -> List[str]:
        names = ["Tbg"]
        for param in ["Tkin", "nH2", "VLSR", "dV", "NCol"]:
            for i in range(self.ncomponent):
                names.append(f"{param}_{i}")
        return names

    def __repr__(self) -> str:
        output = f"Model: {type(self).__name__}\n"
        for dist in self._distributions:
            output += f"{dist}\n"
        return output

    def sample_prior(self) -> np.ndarray:
        """
        Draw samples from each respective prior distribution to
        return an array of parameters.

        Returns
        -------
        np.ndarray
            NumPy 1D array of parameter values drawn from the
            respective prior.
        """
        initial = np.array([param.sample() for param in self._distributions])
        return initial

    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        """
        Wraps `molsim` functionality to simulate the spectrum, given a set
        of input parameters as a NumPy 1D array.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameters for the simulation.

        Returns
        -------
        np.ndarray
            NumPy 1D array corresponding to the simulated spectrum
        """
        Tbg = parameters[0]
        Tkins = parameters[1:1+self.ncomponent]
        nH2s = parameters[1+self.ncomponent:1+self.ncomponent*2]
        vlsrs = parameters[1+self.ncomponent*2:1+self.ncomponent*3]
        dVs = parameters[1+self.ncomponent*3:1+self.ncomponent*4]
        Ncols = parameters[1+self.ncomponent*4:1+self.ncomponent*5]
        for i in range(self.ncomponent):
            if nH2s[i] < 1e3:
                nH2s[i] = 10 ** nH2s[i]
            if Ncols[i] < 1e3:
                Ncols[i] = 10 ** Ncols[i]

        simulated = np.zeros_like(self.observation.spectrum.frequency)
        for Tkin, nH2, vlsr, dV, Ncol, source_size in zip(Tkins, nH2s, vlsrs, dVs, Ncols, self.source_sizes):
            source = NonLTESource(
                velocity=vlsr,
                dV=dV,
                column=Ncol,
                Tbg=Tbg,
                Tkin=Tkin,
                collision_density={'H2': nH2},
                collision_file=self.collision_file,
                radex_output=self.radex_output
            )
            sim = MaserSimulation(
                observation=self.observation,
                source=[source],
                continuum=self.continuum,
                size=source_size,
                units='Jy/beam',
                use_obs=True,
                aperture=self.aperture
            )
            simulated += sim.spectrum.int_profile
        return simulated


    def prior_constraint(self, parameters: np.ndarray) -> float:
        """
        Function that will apply a constrain on the prior. This function
        should be overwritten in child models, say for example in the
        TMC-1 four component case, where we want to constrain parameter
        space to certain regions.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameter values

        Returns
        -------
        float
            Return zero if parameters pass the constraint, otherwise
            return -np.inf
        """
        return 0.0

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        """
        Calculate the total prior log likelihood. The calculation is handed
        off to the individual distributions.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing the model parameters

        Returns
        -------
        float
            The total prior log likelihood
        """
        lnlikelihood = self.prior_constraint(parameters)
        lnlikelihood += sum(
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
            Log likelihood of the model
        """
        obs = self.observation.spectrum
        simulation = self.simulate_spectrum(parameters)
        # match the simulation with the spectrum
        lnlike = - np.log(np.sqrt(2 * np.pi)) * simulation.size - np.sum( np.log(np.fabs(obs.noise)) ) - 0.5 * np.sum( ((obs.Iv - simulation) / obs.noise)**2.0 )
        return lnlike

    def nll(self, parameters: np.ndarray) -> float:
        """
        Calculate the negative log likelihood. This is functionally exactly
        the sample as `compute_log_likelihood`, except that the sign of the
        likelihood is negative for use in maximum likelihood estimation.

        Parameters
        ----------
        parameters : np.ndarray
            [description]

        Returns
        -------
        float
            Negative log likelihood of the model
        """
        return -self.compute_log_likelihood(parameters)

    def mle_optimization(
        self,
        initial: Union[None, np.ndarray] = None,
        bounds: Union[None, List[Union[Tuple[float], float]], None] = None,
        **kwargs,
    ):
        """
        Obtain a maximum likelihood estimate, given an initial starting point in
        parameter space. Because of the often highly covariant nature of models,

        Additional kwargs are passed into `scipy.optimize.minimize`, and can be
        used to overwrite things like the optimization method.

        The `Result` object from `scipy.optimize` is returned, which holds the
        MLE parameters as the attribute `x`, and the likelihood value as `fun`.

        Parameters
        ----------
        initial : Union[None, np.ndarray], optional
            Initial parameters for optimization, by default None, which
            will take the mean of the prior.
        bounds : Union[None, List[Union[Tuple[float, float]]], None], optional
            Bounds for constrained optimization. By default None, which
            imposes no constraints (highly not recommended!). See the
            `scipy.optimize.minimize` page for how `bounds` is specified.

        Returns
        -------
        `scipy.optimize.Result`
            A fit `Result` object that contains the final state of the
            minimization
        """
        if initial is None:
            initial = np.array([self.sample_prior() for _ in range(3000)]).mean(axis=0)
        opt_kwargs = {
            "fun": self.nll,
            "x0": initial,
            "method": "Powell",
            "bounds": bounds,
        }
        opt_kwargs.update(**kwargs)
        result = minimize(**opt_kwargs)
        return result

    @classmethod
    def from_yml(cls, yml_path: str):
        input_dict = load_yaml(yml_path)
        cls_dict = dict()
        # the two stragglers
        for key in input_dict.keys():
            if key not in ["observation", "molecule", "nominal_vlsr"]:
                if hasattr(input_dict[key], "mu"):
                    dist = GaussianLikelihood
                else:
                    dist = UniformLikelihood
                cls_dict[key] = dist.from_values(**input_dict[key])
            else:
                if key != "nominal_vlsr":
                    # load in the observed data
                    cls_dict[key] = load(input_dict[key])
                else:
                    logger.warning(f"{key} is not recognized, and therefore ignored.")
        return cls(**cls_dict)
