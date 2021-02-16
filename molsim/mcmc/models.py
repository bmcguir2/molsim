from typing import Type, Dict, Union, Callable, List, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numexpr as ne
from loguru import logger
from joblib import load
from scipy.optimize import minimize

from molsim.mcmc.base import (
    AbstractModel,
    AbstractDistribution,
    UniformLikelihood,
    GaussianLikelihood,
    DeltaLikelihood,
)
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

    def get_names(self) -> List[str]:
        return ["SourceSize", "VLSR", "NCol", "Tex", "dV"]

    def __repr__(self) -> str:
        output = f"Model: {type(self).__name__}\n"
        for dist in self._distributions:
            output += f"{dist}\n"
        return output

    def sample_prior(self):
        initial = np.array([param.sample() for param in self._distributions])
        return initial

    def simulate_spectrum(
        self, parameters: np.ndarray, scale: float = 3.0
    ) -> np.ndarray:
        """
        Wraps `molsim` functionality to simulate the spectrum, given a set
        of input parameters as a NumPy 1D array. On the first pass, this generates
        a `Simulation` instance and stores it, which has some overhead associated
        with figuring out which catalog entries to simulate. After the first
        pass, the instance is re-used with the `Source` object updated with
        the new parameters.
        
        The nuance in this function is with `scale`: during the preprocess
        step, we assume that the observation frequency is not shifted to the
        source reference. To simulate with molsim, we identify where the catalog
        overlaps with our frequency windows, and because it is unshifted this
        causes molsim to potentially ignore a lot of lines (particularly 
        high frequency ones). The `scale` parameter scales the input VLSR
        as to make sure that we cover everything as best as we can.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameters for the simulation.
        scale : float, optional
            Modifies the window to consider catalog overlap, by default 3.

        Returns
        -------
        np.ndarray
            NumPy 1D array corresponding to the simulated spectrum
        """
        size, vlsr, ncol, Tex, dV = parameters
        # Assume that the value is in log space, if it's below 1000
        if ncol <= 1e3:
            ncol = 10**ncol
        source = Source("", vlsr, size, column=ncol, Tex=Tex, dV=dV)
        if not hasattr(self, "simulation"):
            min_freq, max_freq = find_limits(self.observation.spectrum.frequency)
            # there's a buffer here just to make sure we don't go out of bounds
            # and suddenly stop simulating lines
            min_offsets = compute.calculate_dopplerwidth_frequency(
                min_freq, vlsr * scale
            )
            max_offsets = compute.calculate_dopplerwidth_frequency(
                max_freq, vlsr * scale
            )
            min_freq -= min_offsets
            max_freq += max_offsets
            self.simulation = Simulation(
                mol=self.molecule,
                ll=min_freq,
                ul=max_freq,
                observation=self.observation,
                source=source,
                line_profile="gaussian",
                use_obs=True,
            )
        else:
            self.simulation.source = source
            self.simulation._apply_voffset()
            self.simulation._calc_tau()
            self.simulation._make_lines()
            self.simulation._beam_correct()
        intensity = self.simulation.spectrum.int_profile
        return intensity

    def prior_constraint(self, parameters: np.ndarray):
        pass

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
            Log likelihood of the model
        """
        obs = self.observation.spectrum
        simulation = self.simulate_spectrum(parameters)
        # match the simulation with the spectrum
        lnlike = np.sum(
            np.log(1.0 / np.sqrt(obs.noise ** 2.0))
            * np.exp(-((obs.Tb - simulation) ** 2.0) / (2.0 * obs.noise ** 2.0))
        )
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
        bounds: Union[None, List[Union[Tuple[float, float]]], None] = None,
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
        molecule: Molecule,
    ):
        super().__init__(source_sizes, vlsrs, Ncols, Tex, dV, observation, molecule)
        self.components = list()
        # these are not used in preference of `self.components` instead
        self.source_size = self.vlsr = self.Ncol = self._distributions = None
        for ss, vlsr, Ncol in zip(source_sizes, vlsrs, Ncols):
            self.components.append(
                SingleComponent(
                    ss, vlsr, Ncol, Tex, dV, observation, molecule,
                )
            )

    def get_names(self):
        names = list()
        n_components = len(self.components)
        for parameter in ["SourceSize", "VLSR", "NCol"]:
            names.extend([parameter + f"_{i}" for i in range(n_components)])
        names.extend(["Tex", "dV"])
        return names

    def __repr__(self) -> str:
        output = f"Model {type(self).__name__}\n"
        for index, component in enumerate(self.components):
            output += f"Component {index + 1}: {component}\n"
        return output

    @property
    def distributions(self):
        dists = list()
        for component in self.components:
            dists.extend(component._distributions)
        return dists

    @classmethod
    def from_yml(cls, yml_path: str):
        input_dict = load_yaml(yml_path)
        cls_dict = dict()
        source_sizes, vlsrs, Ncols = list(), list(), list()
        # make sure the number of components is the same
        assert (
            len(input_dict["source_sizes"])
            == len(input_dict["vlsrs"])
            == len(input_dict["Ncols"])
        )
        n_components = len(input_dict["source_sizes"])
        # parse in all the different parameters
        for param_list, parameter in zip(
            [source_sizes, vlsrs, Ncols], ["source_sizes", "vlsrs", "Ncols"]
        ):
            for index in range(n_components):
                size_params = input_dict[parameter][index]
                size_params["name"] = f"{parameter}_{index}"
                if "mu" in size_params:
                    dist = GaussianLikelihood
                elif "value" in size_params:
                    dist = DeltaLikelihood
                else:
                    dist = UniformLikelihood
                param_list.append(dist.from_values(**size_params))
            cls_dict[parameter] = param_list
        # the two stragglers
        for key in ["Tex", "dV"]:
            input_dict[key]["name"] = key
            if "mu" in input_dict[key]:
                dist = GaussianLikelihood
            elif "value" in input_dict[key]:
                dist = DeltaLikelihood
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

    def sample_prior(self):
        source_sizes = list()
        vlsrs = list()
        ncols = list()
        for index, component in enumerate(self.components):
            values = component.sample_prior()
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

    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
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


class TMC1FourComponent(MultiComponent):
    def __init__(
        self,
        source_sizes: List[AbstractDistribution],
        vlsrs: List[AbstractDistribution],
        Ncols: List[AbstractDistribution],
        Tex: AbstractDistribution,
        dV: AbstractDistribution,
        observation: Observation,
        molecule: Molecule,
    ):
        super().__init__(
            source_sizes, vlsrs, Ncols, Tex, dV, observation, molecule,
        )

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        vlsr1, vlsr2, vlsr3, vlsr4 = parameters[[4, 5, 6, 7]]
        if (
            (vlsr1 < (vlsr2 - 0.05))
            and (vlsr2 < (vlsr3 - 0.05))
            and (vlsr3 < (vlsr4 - 0.05))
            and (vlsr2 < (vlsr1 + 0.3))
            and (vlsr3 < (vlsr2 + 0.3))
            and (vlsr4 < (vlsr3 + 0.3))
        ):
            for index, component in enumerate(self.components):
                subparams = self._get_component_parameters(parameters, index)
                if index == 0:
                    lnlikelihood = component.compute_prior_likelihood(subparams)
                else:
                    lnlikelihood += component.compute_prior_likelihood(subparams)
            return lnlikelihood
        return -np.inf
