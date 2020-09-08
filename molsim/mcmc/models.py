from typing import Type, Dict
from pathlib import Path

import pymc3 as pm
import numpy as np
from loguru import logger

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
