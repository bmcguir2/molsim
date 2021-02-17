from typing import Type, Dict, Union, Callable, List, Tuple, Iterable
from functools import lru_cache
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
        return 0.

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

    def prior_constraint(self, parameters: np.ndarray) -> float:
        """
        Applies a constraint on the model parameters as a whole.
        The idea is that for more complex models, this method should
        be overridden, rather than having to redefine the likelihood
        calculation itself.
        
        Another way of thinking about this would be to use decorators,
        but that may be more complicated than what is needed here.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameter values

        Returns
        -------
        float
            Returns 0 because no constraint is applied
        """
        return 0.

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        lnlikelihood = self.prior_constraint(parameters)
        for index, component in enumerate(self.components):
            subparams = self._get_component_parameters(parameters, index)
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

    def prior_constraint(self, parameters: np.ndarray) -> float:
        """
        Applies the TMC-1 four component velocity constraint, where we
        make sure that the four velocity components do not stray too
        far from one another.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameter values

        Returns
        -------
        float
            Return zero if the constraints are met, otherwise negative
            infinity.
        """
        vlsr1, vlsr2, vlsr3, vlsr4 = parameters[[4, 5, 6, 7]]
        if (
            (vlsr1 < (vlsr2 - 0.05))
            and (vlsr2 < (vlsr3 - 0.05))
            and (vlsr3 < (vlsr4 - 0.05))
            and (vlsr2 < (vlsr1 + 0.3))
            and (vlsr3 < (vlsr2 + 0.3))
            and (vlsr4 < (vlsr3 + 0.3))
        ):
            return 0.
        else:
            return -np.inf


class CospatialTMC1(TMC1FourComponent):
    """
    Implementation of a multi component model. This type of model extends
    the parameters expected with parameters for each component like so:
    
    [source_size, vlsr1, vlsr2,...Tex, dV]
    
    So that there is an arbitrary number of components, providing each
    component has a common source size, and independent radial velocity
    and column density.
    """

    def __init__(
        self,
        source_size: AbstractDistribution,
        vlsrs: List[AbstractDistribution],
        Ncols: List[AbstractDistribution],
        Tex: AbstractDistribution,
        dV: AbstractDistribution,
        observation: Observation,
        molecule: Molecule,
    ):
        # make four source sizes drawn from the same distribution
        source_sizes = [source_size] * len(vlsrs)
        # initialize the base class
        super().__init__(source_sizes, vlsrs, Ncols, Tex, dV, observation, molecule)

    def get_names(self):
        names = ["SourceSize"]
        n_components = len(self.components)
        for parameter in ["VLSR", "NCol"]:
            names.extend([parameter + f"_{i}" for i in range(n_components)])
        names.extend(["Tex", "dV"])
        return names

    @classmethod
    def from_yml(cls, yml_path: str):
        input_dict = load_yaml(yml_path)
        cls_dict = dict()
        vlsrs, Ncols = list(), list()
        # make sure the number of components is the same
        assert (
            len(input_dict["vlsrs"])
            == len(input_dict["Ncols"])
        )
        n_components = len(input_dict["vlsrs"])
        # parse in all the different parameters
        for param_list, parameter in zip(
            [vlsrs, Ncols], ["vlsrs", "Ncols"]
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
        # the three stragglers
        for key in ["source_size", "Tex", "dV"]:
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
        # VLSR and NCol for each source, shared SS, Tex, and dv
        return len(self.components) * 2 + 3

    def sample_prior(self):
        vlsrs = list()
        ncols = list()
        for index, component in enumerate(self.components):
            # grab values from the prior
            values = component.sample_prior()
            vlsrs.append(values[1])
            ncols.append(values[2])
            if index != len(self.components):
                final = values[-2:]
        # get the source size from only one distribution
        params = [self.components[0].sample_prior()[0]]
        for array in [vlsrs, ncols, final]:
            params.extend(array)
        return params

    def _get_component_parameters(
        self, parameters: np.ndarray, component: int
    ) -> np.ndarray:
        """
        Get the parameters of each component. This is somewhat ugly as the number of
        parameters is fewer than the usual four component model.
        """
        subparams = np.asarray([
            parameters[0], parameters[component + 1], parameters[component + (1 + 4)],
            parameters[-2], parameters[-1]
            ])
        return subparams

    def prior_constraint(self, parameters: np.ndarray) -> float:
        """
        Applies the TMC-1 four component velocity constraint, where we
        make sure that the four velocity components do not stray too
        far from one another. This is modified to match the correct
        indices for the parameters (given there is only one source size).
        
        Not the cleanest way to do this, but probably the most straightforward.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameter values

        Returns
        -------
        float
            Return zero if the constraints are met, otherwise negative
            infinity.
        """
        vlsr1, vlsr2, vlsr3, vlsr4 = parameters[[1, 2, 3, 4]]
        if (
            (vlsr1 < (vlsr2 - 0.05))
            and (vlsr2 < (vlsr3 - 0.05))
            and (vlsr3 < (vlsr4 - 0.05))
            and (vlsr2 < (vlsr1 + 0.3))
            and (vlsr3 < (vlsr2 + 0.3))
            and (vlsr4 < (vlsr3 + 0.3))
        ):
            return 0.
        else:
            return -np.inf


class CompositeModel(AbstractModel):
    """
    Implements an abstract composite model. Using this by
    itself will not work as parameter sharing is not implemented
    correctly, but basically lays out the recipe for implementing
    real composite models.
    """
    def __init__(self, param_indices: List[List[int]], *models, **kwargs) -> None:
        """
        Parameters
        ----------
        param_indices : List[List[int]]
            Nested list of integers corresponding to the
            parameter indices of each model. The top list
            should be the same length as the number of models,
            and each sublist should have the same number of
            indices as expected by each model.
        """
        assert len(param_indices) == len(list(models))
        super().__init__()
        self.models = list(models)
        self.param_indices = param_indices

    def __len__(self) -> int:
        # do a messy flattened list comprehension and get the largest
        # index out, plus one for the number of parameters
        return max([index for sublist in self.param_indices for index in sublist]) + 1

    def __repr__(self):
        output_string = f"Composite model\nNumber of parameters: {len(self)}\n"
        for model in self.models:
            output_string += f"Model: {model}\n"
        return output_string

    def model_parameter(self, parameters: np.ndarray, index: int) -> np.ndarray:
        """
        Get the model parameters associated with a particular submodel.
        All this function does is cross reference the model index with
        the `param_indices` attribute, and returns the parameter values
        associated with that particular model.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing the full parameter values
        index : int
            Model index number

        Returns
        -------
        np.ndarray
            NumPy 1D array containing the parameters
            for the specified submodel
        """
        return parameters[self.param_indices[index]]

    @lru_cache(maxsize=2, typed=True)
    @property
    def frequency(self) -> np.ndarray:
        """
        Returns a common frequency grid for all the models considered. This allows
        for a combined simulation to be performed.
        
        This property should also be cached, as to prevent constant recalculation
        of the frequency grid.

        Returns
        -------
        np.ndarray
            NumPy 1D array of frequencies
        """
        frequency = [model.observation.spectrum.frequency for model in self.models]
        # get unique values of frequency and then sort
        frequency = np.unique(np.concatenate(frequency))
        frequency.sort()
        return frequency

    @lru_cache(maxsize=2, typed=True)
    def simulate_spectrum(self, parameters: np.ndarray) -> np.ndarray:
        """
        Simulate the composite spectrum for multiple models. This uses
        the `param_indices` attribute to properly allocate parameters
        to each model.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing all of the model parameters.

        Returns
        -------
        np.ndarray
            NumPy 1D array containing the simulated intensities combined
            from each model.
        """
        spectrum = np.zeros_like(self.frequency)
        for index, model in enumerate(self.models):
            # retrieve the correct parameters
            subparams = self.model_parameter(parameters, index)
            temp_spectrum = model.simulate_spectrum(subparams)
            # add the interpolated spectrum
            spectrum += np.interp(self.frequency, model.observation.spectrum.frequency, temp_spectrum, left=0., right=0.)
        return spectrum

    def compute_prior_likelihood(self, parameters: np.ndarray) -> float:
        """
        Compute the composite prior likelihood, combined from all of
        the submodels. Note that this function applies a top-level
        prior constraint implemented in `CompositeModel`, in addition
        to the submodel constraints.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing the full sampled parameters

        Returns
        -------
        float
            Prior log likelihood summed over all submodels
        """
        likelihood = self.prior_constraint(parameters)
        for index, model in enumerate(self.models):
            subparams = self.model_parameter(parameters, index)
            likelihood += model.compute_prior_likelihood(subparams)
        return likelihood

    def compute_log_likelihood(self, parameters: np.ndarray) -> float:
        """
        Compute the composite log likelihood, combined from all of
        the models.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing the full sampled parameters

        Returns
        -------
        float
            Log likelihood summed over all submodels
        """
        likelihood = sum([model.compute_log_likelihood(parameters[indices]) for indices, model in zip(self.param_indices, self.models)])
        return likelihood

    def prior_constraint(self, parameters: np.ndarray) -> float:
        """
        Apply a composite model constraint. Because of how the models
        are implemented, the submodels can have their own independent
        prior constraints, and this is applied prior to those.

        Parameters
        ----------
        parameters : np.ndarray
            NumPy 1D array containing parameters
        """
        return 0.

    def sample_prior(self) -> np.ndarray:
        """
        Sample from the priors of each submodel, and ensure that the
        number of parameters that come up match what is actually
        expected.

        Returns
        -------
        np.ndarray
            NumPy 1D array with parameter values
        """
        parameters = np.zeros(len(self))
        # loop over each model, sample from their prior, and organize those
        # parameters according to the `param_indices` dictate
        for index, model in enumerate(self.models):
            param_indices = self.param_indices[index]
            subparams = model.sample_prior()
            # loop over each subparameter, and allocate to the correct parameter element
            for param_index, subparam in zip(param_indices, subparams):
                parameters[param_index] = subparam
        return parameters


class TMC1MethylChains(CompositeModel):
    """
    Single source, shared velocity components, shared linewidth,
    different Ncol and Tex.
    
    [ss, vlsr1, vlsr2, vlsr3, vlsr4, NcolA_1, NcolA_2, NcolA_3, NcolA_4,
    NcolB_1, NcolB_2, NcolB_3, NcolB_4, TexA, TexB, dv
    ]
    
    16 parameters :P
    
    The idea here would be to just have the composite model do the
    managing, and just pass the correct parameters to the correct
    model; everything else behaves the same as they would otherwise.
    All this child model of `CompositeModel` does is prescribe the
    parameter indices.
    
    `A_state` and `E_state` are expected to be instances of `Cospatial`
    models.
    """
    def __init__(self, A_state, E_state, **kwargs) -> None:
        param_indices = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 15],
            [0, 1, 2, 3, 4, 9, 10, 11, 12, 14, 15]
        ]
        super().__init__(param_indices, A_state, E_state, **kwargs)

