from typing import Type, Any, List, Union, Tuple, Callable

import numpy as np
import numexpr as ne
from loguru import logger
from numba import jit, njit, prange, config
from tqdm.auto import tqdm

from molsim.classes import Catalog
from molsim.constants import cm, kcm, ccm, h, k, ckm
from molsim import utils


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


def calc_noise_std(intensity, threshold=3.5) -> Tuple[np.ndarray, np.ndarray]:
    dummy_ints = np.copy(intensity)
    noise = np.copy(intensity)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # repeats 3 times to make sure to avoid any interloping lines
    for chan in np.where(dummy_ints < (-dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints < (-noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints < (-dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    return noise_mean, noise_std


def calculate_tau(
    catalog: Type["Catalog"], Ncol: float, Q: float, Tex: float, dV: float,
) -> np.ndarray:
    """
    Compute the optical depth based on theoretical intensities, the partition
    function, excitation temperature, and radial velocity.

    Parameters
    ----------
    catalog : Type[
        Molsim `Catalog` object instance, containing the intrinsic linestrength,
        the upper state degeneracy and energy, and the frequency of each line.
    Ncol : float
        Column density in cm^-2 of the molecule
    Q : float
        Partition function at Tex
    Tex : float
        Excitation temperature in K
    dV : float
        Radial velocity in km/s

    Returns
    -------
    np.ndarray
        Optical depth tau predicted for each catalog entry
    """
    gup, eup, linestrength, frequency = (
        catalog.gup,
        catalog.eup,
        catalog.aij,
        catalog.frequency,
    )
    mask = catalog.mask
    tau = (
        linestrength[mask] * cm ** 3 * (Ncol * 100**2) * gup[mask]
        * (np.exp(-eup[mask] / Tex))
        * (np.exp(h * frequency[mask] * 1e6 / (k * Tex)) - 1)
    ) / (8 * np.pi * (frequency[mask] * 1e6) ** 3 * dV * 1000 * Q)
    return tau


def calculate_background(
    temperature: float, frequency: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Tbg = np.full_like(frequency, temperature)
    return Tbg


def calculate_Iv(tau: np.ndarray, frequency: np.ndarray, Tex: float) -> np.ndarray:
    """
    Calculate the flux/brightness of a given line; the `frequency` array should
    correspond to catalog entries, and the `tau` array should be the optical
    depth at each frequency.

    Parameters
    ----------
    tau : np.ndarray
        [description]
    frequency : np.ndarray
        [description]
    Tex : float
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    Iv = (
        (2 * h * tau * (frequency * 1e6) ** 3)
        / cm ** 2
        * (np.exp(h * frequency * 1e6 / (k * Tex)) - 1)
    ) * 1e26
    return Iv


def continuum_tau_correction(
    frequency: np.ndarray, tau: np.ndarray, Tbg: np.ndarray, Tex: float
):
    """
    Performs inplace correction of the calculated optical depth with the
    background continuum.

    Parameters
    ----------
    frequency : np.ndarray
        [description]
    tau : np.ndarray
        [description]
    Tbg : np.ndarray
        [description]
    Tex : float
        [description]
    """
    J_T = (h * frequency * 1e6 / k) * (
        np.exp(((h * frequency * 1e6) / (k * Tex))) - 1
    ) ** -1
    J_Tbg = (h * frequency * 1e6 / k) * (
        np.exp(((h * frequency * 1e6) / (k * Tbg))) - 1
    ) ** -1
    return (J_T - J_Tbg) * (1 - np.exp(-tau))


def atomic_gaussian(
    obs_frequency: np.ndarray, centers: np.ndarray, tau: np.ndarray, dV: float,
):
    n_catalog = centers.size
    intensities = np.zeros_like(obs_frequency)
    for index in range(n_catalog):
        intensities += neu_gaussian(
            obs_frequency, centers[index], tau[index], dV
        )
    return intensities


@njit
def neu_gaussian(x, x0, A, dV):
    return A * np.exp(-(x - x0)**2. / (2*((dV*x0/ckm)/2.35482)**2))


def beam_correction(
    frequency: np.ndarray, intensity: np.ndarray, source_size: float, dish_size: float
):
    """
    Correct the simulated lineprofiles inplace for discrepancy between the
    telescope beam and the source size.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency values of each obs channel
    intensity : np.ndarray
        Line profile values of each obs channel
    source_size : float
        Size of the source
    dish_size : float
        Size of the dish
    """
    # convert beam size to arcsec
    beam_size = 206265 * 1.22 * (cm / (frequency * 1e6)) / dish_size
    intensity *= source_size ** 2 / (beam_size ** 2 + source_size ** 2)


def build_synthetic_spectrum(
    source_size: float,
    vlsr: float,
    Ncol: float,
    Tex: float,
    dV: float,
    spectrum: Type["DataChunk"],
    catalog: Type["Catalog"],
    dish_size: float,
    calc_Q: Union[Callable, str],
    background_temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Velocity offset
    2. Calculate optical depth
    3. Calculate background
    4. Calculate optical depth correction
    5. Simulate line profiles
    6. Apply beam dilution correction

    Parameters
    ----------
    spectrum : Type[
        [description]
    catalog : Type[
        [description]
    vlsr : float
        [description]
    source_size : float
        [description]
    dish_size : float
        [description]
    Ncol : float
        [description]
    Tex : float
        [description]
    dV : float
        [description]
    background_temperature : float
        [description]
    n_chunks : int, optional
        [description], by default 200

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [description]
    """
    # set the numba threads for parallelism, used mainly for the
    # lineshape calculation
    # config.set_num_threads(max_threads)
    if not hasattr(catalog, "mask") or spectrum.mask is not None:
        catalog.mask = spectrum.mask
    else:
        catalog.mask = np.ones_like(catalog.frequency, dtype=bool)
    # calculate Q
    if type(calc_Q) == str:
        Q = ne.evaluate(calc_Q, local_dict={"Tex": Tex})
    elif callable(calc_Q):
        Q = calc_Q(Tex)
    else:
        raise NotImplementedError("`calc_Q` arg is invalid; must be callable function or string.")
    offset_freq = utils._apply_vlsr(spectrum.frequency, vlsr)
    masked_freqs = catalog.frequency[catalog.mask]
    # calculate continuum background as just a flat array of temperature
    Tbg = np.full_like(masked_freqs, background_temperature)
    Tbg_full = np.full_like(spectrum.frequency, background_temperature)
    tau = calculate_tau(catalog, Ncol, Q, Tex, dV)
    # this corrects tau inplace
    # continuum_tau_correction(masked_freqs, tau, Tbg, Tex)
    # compute line shapes
    sim_int = atomic_gaussian(spectrum.frequency, masked_freqs, tau, dV)
    continuum_tau_correction(offset_freq, sim_int, Tbg_full, Tex)
    # apply beam dilution
    sim_int = utils._apply_beam(offset_freq, sim_int, source_size, dish_size)
    # beam_correction(offset_freq, sim_int, source_size, dish_size)
    return offset_freq, sim_int#, masked_freqs, tau

