from typing import Type, Dict
from pathlib import Path

import pymc3 as pm
import numpy as np
from loguru import logger

from molsim.mcmc.base import BaseHelper
from molsim.mcmc import compute, preprocess
from molsim.file_handling import load_mol
from molsim.classes import Molecule


class TMC1_FourComponent(pm.Model):
    def __init__(
        self,
        frequency: np.ndarray,
        intensity: np.ndarray,
        name="TMC1_FourComponent",
        model=None,
    ):
        super().__init__(name=name, model=model)
        Tex = pm.Uniform("Tex", 0.0, 10.0)
        dV = pm.Uniform("dV", 0.0, 3.0)
        # define priors for the four sources
        parameters = list()
        for source_index in range(4):
            prefix = f"Source{source_index}"
            sizes.append(pm.Uniform(f"{prefix}_size", 0.0, 400.0))
            vlsrs.append(pm.Uniform(f"{prefix}_vlsr", 0.0, 3.0))
            ncols.append(pm.Uniform(f"{prefix}_ncol", 0.0, 1e16))
        # compute the model spectrum
        # model_spectrum = predict()
        # the observed intensity as a Gaussian likelihood
        # pm.Normal("Y_obs", mu=model_spectrum)


class TMC1Helper(BaseHelper):
    def __init__(self, target: Type[preprocess.DataChunk], molecule: Type[Molecule]):
        super().__init__()
        self._target = target
        self._molecule = molecule
        self._model = TMC1_FourComponent(self._target.frequency, self._target.intensity)

    def build_model(self):
        pass

    def model(self):
        return self._model

    def target(self):
        return self._target

    def molecule(self):
        return self._molecule

    def traces(self):
        return self._traces

    @classmethod
    def from_dict(cls, param_dict: Dict[str, str]):
        logger.add("LOG")
        for key in ["catalog", "spectrum_path", "delta_v"]:
            if key not in param_dict:
                raise KeyError(f"{key} entry is missing from input dictionary!")
        logger.info("Loading molecule from SPCAT format.")
        molecule = load_mol(param_dict.get("catalog"), type="SPCAT")
        logger.info("Loading and preprocessing spectrum.")
        spectrum = preprocess.load_spectrum(
            param_dict.get("spectrum_path"),
            molecule.catalog,
            param_dict.get("delta_v"),
            param_dict.get("rbf_params", {}),
            param_dict.get("noise_params", {}),
            param_dict.get("n_workers", 4)
            )
        return cls(spectrum, molecule)

