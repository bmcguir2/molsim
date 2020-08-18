from typing import Type, Any, List, Union

import numpy as np
from loguru import logger
from numba import jit, njit

from molsim.classes import Catalog
from molsim.constants import cm, kcm, ccm, h, k, ckm


def calculate_dopplerwidth_frequency(
    frequencies: np.ndarray, delta_v: float
) -> np.ndarray:
    """
    Analytic expression to 

    Parameters
    ----------
    frequencies : np.ndarray
        [description]
    delta_v : float
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    return np.abs(frequencies * delta_v / (ckm * 2.0 * np.log(2.0)))


@njit(fastmath=True)
def apply_beam(
    frequencies: np.ndarray,
    intensities: np.ndarray,
    source_size: float,
    dish_size: float,
):
    """
    Compute beam corrections to the intensity. This offsets the "effective"
    observed intensity based on the size of the telescope beam and the source.

    Parameters
    ----------
    frequency : np.ndarray
        NumPy 1D 
    intensity : np.ndarray
        [description]
    source_size : float
        [description]
    dish_size : float
        [description]

    Returns
    -------
    np.ndarray
        Corrected intensities
    """
    # create a wave to hold wavelengths, fill it to start w/ frequencies
    wavelength = cm / (frequencies * 1e6)
    # fill it with beamsizes
    beam_size = wavelength * 206265 * 1.22 / dish_size
    # create an array to hold beam dilution factors
    dilution_factor = source_size ** 2 / (beam_size ** 2 + source_size ** 2)
    # perform an inplace modification to the intensities
    intensities *= dilution_factor
    return None


def predict_spectrum(
    data: np.ndarray,
    source_size: float,
    Ncol: float,
    Tex: float,
    dV: float,
    Q: float,
    mol_cat: Type[Catalog],
    obs,
):
    """
    TODO - make this work lol

    Parameters
    ----------
    data : np.ndarray
        [description]
    source_size : float
        [description]
    Ncol : float
        [description]
    Tex : float
        [description]
    dV : float
        [description]
    Q : float
        [description]
    mol_cat : Type[Catalog]
        [description]
    obs : [type]
        [description]
    """
    spec_frequency = data[:, 0].view()
    spec_int = data[:, 1].view()
    # update the Einstein A coefficients with given partition function
    mol_cat._set_sijmu_aij(Q)
    Nl = Ncol * mol_cat.glow * np.exp(-mol_cat.elow / (kcm * Tex)) / Q
    # calculate the optical depth
    tau = (
        (ccm / (mol_cat.frequency * 10 ** 6)) ** 2
        * mol_cat.aij
        * mol_cat.gup
        * Nl
        * (1 - np.exp(-(h * mol_cat.frequency * 10 ** 6) / (k * T)))
    )
    # no re-allocation needed
    tau /= 8 * np.pi * (dV * mol_cat.frequency * 10 ** 6 / ckm) * mol_cat.glow
    # calculate the simulated spectrum
    int_sim = lineshapes.simulate_gaussians(
        spec_frequency,
        mol_cat.intensity,
        mol_cat.frequency,
        [dV for _ in range(mol_cat.frequency.size)],
    )
    # apply beam dilution correction to intensities inplace
    apply_beam(
        spec_frequency, int_sim, source_size,
    )


def simulate_gaussians(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    centers: np.ndarray,
    linewidths: np.ndarray,
) -> np.ndarray:
    """
    Calculate the flattened spectrum for a mixture of Gaussian
    lineshapes. This function takes heavy advantage of broadcasting,
    which makes the code pretty optimized. 

    Parameters
    ----------
    frequencies : np.ndarray
        [description]
    amplitudes : np.ndarray
        [description]
    centers : np.ndarray
        [description]
    linewidths : np.ndarray
        [description]

    Returns
    -------
    np.ndarray
        A NumPy 1D array containing the composite spectrum
    """
    assert amplitudes.size == centers.size == linewidths.size
    intensities = np.exp(
        -((centers[:, np.newaxis] - frequencies) ** 2.0)
        / (2.0 * linewidths[:, np.newaxis] ** 2.0)
    )
    # add amplitudes in place
    intensities *= amplitudes[:, np.newaxis]
    return intensities.sum(axis=0)
