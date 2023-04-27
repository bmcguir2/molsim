import os
from typing import Type, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from itertools import repeat

import numpy as np
import h5py
import numba
from dask import array as da
from loguru import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from joblib import dump

from molsim.classes import Catalog, Spectrum, Observation
from molsim.mcmc import compute
from molsim.file_handling import _load_catalog


@dataclass
class DataChunk:
    frequency: np.ndarray
    intensity: np.ndarray
    catalog_index: int
    # technically we could instantiate with a NumPy array,
    # but None is probably faster
    noise: Any = None
    mask: Any = None

    def __len__(self):
        return self.frequency.size

    def __repr__(self):
        return f"DataChunk with {len(self)} elements. Noise: {self.noise is not None}, Mask: {self.mask is not None}"

    def to_hdf5(self, filename: str, **kwargs):
        with h5py.File(filename, mode="a", **kwargs) as h5_file:
            h5_file.create_dataset("frequency", data=self.frequency)
            h5_file.create_dataset("intensity", data=self.intensity)
            h5_file.create_dataset("catalog_index", data=self.catalog_index)
            if self.noise is not None:
                h5_file.create_dataset("noise", data=self.noise)
            if self.mask is not None:
                h5_file.create_dataset("mask", data=self.mask)

    def to_pickle(self, filename: str, **kwargs) -> None:
        dump(self, filename + ".pkl", **kwargs)

    @classmethod
    def from_hdf5(cls, filename: str, dask=False, **kwargs):
        h5_file = h5py.File(filename, **kwargs)
        if dask:
            load_func = da.from_array
        else:
            load_func = np.array
        frequency = load_func(h5_file["frequency"])
        intensity = load_func(h5_file["intensity"])
        catalog_index = load_func(h5_file["catalog_index"])
        if "noise" in h5_file:
            noise = load_func(h5_file["noise"])
        else:
            noise = None
        if "mask" in h5_file:
            mask = load_func(h5_file["mask"])
        return cls(frequency, intensity, catalog_index, noise, mask)

    def to_spectrum(self) -> Spectrum:
        spectrum = Spectrum(frequency=self.frequency, Tb=self.intensity,)
        spectrum.noise = self.noise
        return spectrum

    def to_observation(self, observatory=None) -> Observation:
        spectrum = self.to_spectrum()
        observation = Observation(spectrum=spectrum, observatory=observatory)
        return observation


def extract_frequency_slice(
    data: np.ndarray, rest_frequency: float, delta: float
) -> np.ndarray:
    """
    Grab a chunk of frequency/intensity data, given

    Parameters
    ----------
    data : np.ndarray
        [description]
    frequency : float
        [description]
    delta : float
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    frequencies = data[:, 0].view()
    # Generate a mask in frequency
    mask = np.logical_and(
        frequencies <= rest_frequency + delta, rest_frequency - delta <= frequencies
    )
    return data[mask, :]


@numba.njit(parallel=True)
def filter_catalog(
    spectrum_frequencies: np.ndarray, catalog_frequencies: np.ndarray, max_dist=1.0
) -> np.ndarray:
    mask = np.zeros_like(catalog_frequencies, dtype=np.uint8)
    n_catalog = catalog_frequencies.size
    # parallelize loop over the catalog frequencies
    for index in numba.prange(n_catalog):
        distance = np.sum(
            np.abs(catalog_frequencies[index] - spectrum_frequencies) <= max_dist
        )
        # if we have overlap in the spectrum, take it out
        if distance != 0:
            mask[index] = 1
    return mask


def extract_chunks(
    data: np.ndarray,
    catalog: Type[Catalog],
    delta_v: float = 5.0,
    vlsr: float = 5.8,
    rbf_params={},
    noise_params={},
    verbose: bool = False,
    block_interlopers: bool = False,
    interloper_threshold: float = 6.0,
):
    """
    Function to extract frequency chunks out of a large NumPy 2D array, where
    chunks coincide with catalog line entries. Each chunk will correspond to
    a number of velocity channels.

    Parameters
    ----------
    data : np.ndarray
        NumPy 2D array, where the columns are frequency and intensity,
        and the rows correspond to bins.
    catalog : Type[Catalog]
        [description]
    """
    # Extract only frequencies within band of the observations
    logger.info("Extracting inband chunks.")
    min_freq, max_freq = data[:, 0].min(), data[:, 0].max()
    # njit'd function to find regions of significant overlap between the spectrum
    # and the catalog entries
    mask = filter_catalog(data[:, 0], catalog.frequency).astype(bool)
    logger.info("Calculated mask.")
    # get indices to track which catalog entry is used
    indices = np.arange(catalog.frequency.size)[mask]
    # vectorized computation of the frequency offsets based on a doppler velocity
    offsets = compute.calculate_dopplerwidth_frequency(catalog.frequency[indices], vlsr)
    # nominal vlsr frequencies
    vlsr_catalog_frequencies = catalog.frequency[indices] + offsets
    inband_freqs = np.vstack([indices, vlsr_catalog_frequencies, offsets]).T
    logger.info(f"There are {mask.sum()} catalog hits.")
    chunks = list()
    last_freq = 0.0
    for inband_data in tqdm(inband_freqs):
        # check to make sure the last frequency doesn't overlap
        if abs(inband_data[1] - last_freq) > compute.calculate_dopplerwidth_frequency(
            inband_data[1], delta_v
        ):
            chunk = _compute_chunks(
                inband_data, data, rbf_params, noise_params, verbose
            )
            last_freq = inband_data[1]
        else:
            pass
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks, indices


def _compute_chunks(
    inband_data: Tuple[int, float, float],
    data: np.ndarray,
    rbf_params={},
    noise_params={},
    verbose=False,
) -> Type[DataChunk]:
    """

    Parameters
    ----------
    data : np.ndarray
        [description]
    inband_data : Tuple[int, float]
        [description]
    offset : float
        [description]
    rbf_params : dict, optional
        [description], by default {}
    noise_params : dict, optional
        [description], by default {}

    Returns
    -------
    Type[DataChunk]
        [description]
    """
    index, rest_freq, offset = inband_data
    masked_data = extract_frequency_slice(data, rest_freq, offset)
    chunk = DataChunk(
        frequency=masked_data[:, 0], intensity=masked_data[:, 1], catalog_index=index,
    )
    chunk.noise = gp_noise_estimation(chunk, rbf_params, noise_params, verbose)
    return chunk


def gp_noise_estimation(
    chunk: Type[DataChunk], rbf_params={}, noise_params={}, verbose=False
) -> np.ndarray:
    """
    Uses a simple Gaussian Process model to perform noise estimation on spectral
    data. A given chunk of the full spectrum is fit with a GP model comprising
    RBF and white noise kernels, where the former explains covariance in intensities
    between channels and the latter models variation in the signal as i.i.d white
    noise. 
    
    The GP model is conditioned to provide a maximum likelihood estimate
    of the data, and depends heavily on the initial parameters. The arguments
    `rbf_params` and `noise_params` allow the user to override defaults for the kernels,
    and may require some tweaking to get the desired behavior.
    
    The objective of this function is to estimate the noise at every point of the
    spectrum, and returns a NumPy 1D array of noise values with the same shape as
    the frequency bins.

    Parameters
    ----------
    chunk : Type[DataChunk]
        [description]
    rbf_params : dict, optional
        [description], by default {}
    noise_params : dict, optional
        [description], by default {}

    Returns
    -------
    np.ndarray
        NumPy 1D array containing the noise at every channel
    """
    freq, intensity = chunk.frequency, chunk.intensity
    # RBF parameters affect how correlated each channel is
    # noise parameters affect the variance in signal explained as normally
    # distributed noise
    rbf_kern = {"length_scale": 5e-1, "length_scale_bounds": (1e-1, 10.0)}
    noise_kern = {"noise_level": 1e-1, "noise_level_bounds": (1e-3, 1.0)}
    rbf_kern.update(**rbf_params)
    noise_kern.update(**noise_params)
    # instantiate the model
    kernel = kernels.RBF(**rbf_kern) + kernels.WhiteKernel(**noise_kern)
    gp_model = GaussianProcessRegressor(kernel, normalize_y=True)
    gp_result = gp_model.fit(freq[:, None], intensity[:, None])
    # reproduce the spectrum with uncertainties
    pred_y, pred_std = gp_result.predict(freq[:, None], return_std=True)
    # log some information about the GP result
    if verbose:
        logger.info(f"GP results for catalog index {chunk.catalog_index}.")
        logger.info(
            f"MSE from GP fit: {mean_squared_error(pred_y.flatten(), intensity):.4f}"
        )
        logger.info(
            f"Marginal log likelihood: {gp_result.log_marginal_likelihood_value_:.4f}"
        )
        logger.info(f"Kernel parameters: {gp_result.kernel_}")
    return pred_std


def unroll_chunks(chunks: List[Type[DataChunk]]) -> Tuple[np.ndarray, List[int]]:
    """
    Takes a list of DataChunks, and unwraps it into a 2D NumPy array, where
    each row corresponds to frequency, intensity, and noise respectively, and
    a list of catalog indices corresponding to which catalog entries are relevant
    to each chunk.

    Parameters
    ----------
    chunks : List[Type[DataChunk]]
        List of DataChunks to be unrolled

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        2-tuple comprising the collected data and the catalog indices of
        each chunk
    """
    frequency = list()
    intensity = list()
    noise = list()
    cat_indices = list()
    for index, chunk in enumerate(chunks):
        frequency.append(chunk.frequency)
        intensity.append(chunk.intensity)
        noise.append(chunk.noise)
        cat_indices.append(chunk.catalog_index)
    frequency = np.hstack(frequency)
    intensity = np.hstack(intensity)
    noise = np.hstack(noise)
    return frequency, intensity, noise, cat_indices


def _legacy_filter_spectrum(
    catalog: Catalog,
    frequency: np.ndarray,
    intensity: np.ndarray,
    vlsr: float = 5.8,
    delta_v: float = 0.3,
    block_interlopers: bool = False,
    interloper_threshold: float = 6.0,
    sim_cutoff: float = 0.1,
    line_wash_threshold: float = 3.5,
):
    sorted_index = np.argsort(frequency)
    frequency = frequency[sorted_index]
    intensity = intensity[sorted_index]
    restfreqs = catalog.frequency
    int_sim = 10 ** catalog.logint
    max_int_sim = int_sim.max()
    logger.info("Thresholding catalog entries based on overlap and intensity.")
    logger.info(f"Intensity cutoff: {sim_cutoff * max_int_sim}")
    # get indices of catalogs that actually fall in the range of the data
    cat_mask = np.where(
        (restfreqs < frequency.max())
        & (restfreqs > frequency.min())
        & (int_sim > sim_cutoff * max_int_sim)
    )[0]
    restfreqs = restfreqs[cat_mask]
    logger.info(f"Min/Max catalog frequencies: {restfreqs.min():.4f},{restfreqs.max():.4f}")
    int_sim[cat_mask]
    catalog_indices = list()
    relevant_freqs = np.zeros_like(frequency)
    relevant_intensity = np.zeros_like(intensity)
    relevant_yerrs = np.zeros_like(intensity)
    ignore_counter = 0
    for catalog_index, restfreq in zip(cat_mask, restfreqs):
        freq_ll = restfreq * (1 - (vlsr + abs(delta_v)) / 300000)
        freq_ul = restfreq * (1 - (vlsr - abs(delta_v)) / 300000)
        index_ll = np.searchsorted(frequency, freq_ll)
        index_ul = np.searchsorted(frequency, freq_ul)
        if index_ll < index_ul:
            mask = slice(index_ll, index_ul)
            noise_mean, noise_std = compute.calc_noise_std(
                intensity[mask], line_wash_threshold
            )
            if np.isnan(noise_mean) or np.isnan(noise_std):
                logger.info(f"NaNs found at {restfreq}")
                continue
            if (
                block_interlopers
                and intensity[mask].max() > interloper_threshold * noise_std
            ):
                logger.info(f"Found interloper at {restfreq}; ignoring.")
                ignore_counter += 1
                continue
            else:
                catalog_indices.append(catalog_index)
                relevant_freqs[mask] = frequency[mask]
                relevant_intensity[mask] = intensity[mask]
                relevant_yerrs[mask] = np.sqrt(
                    noise_std ** 2.0 + (intensity[mask] * 0.1) ** 2.0
                )
    logger.info(
        f"Ignored a total of {ignore_counter} catalog entries due to interlopers."
    )
    mask = relevant_freqs > 0
    relevant_freqs = relevant_freqs[mask]
    relevant_intensity = relevant_intensity[mask]
    relevant_yerrs = relevant_yerrs[mask]
    mask = np.zeros_like(catalog.frequency)
    mask[catalog_indices] = 1
    chunk = DataChunk(
        frequency=relevant_freqs,
        intensity=relevant_intensity,
        noise=relevant_yerrs,
        catalog_index=catalog_indices,
        mask=mask.astype(bool),
    )
    return chunk


def preprocess_spectrum(
    name: str,
    spectrum_path: str,
    catalog_path: str,
    delta_v: float,
    vlsr: float,
    rbf_params={},
    noise_params={},
    freq_range: Tuple[float, float] = (0.0, np.inf),
    block_interlopers: bool = False,
    interloper_threshold: float = 6.0,
    observatory=None,
    legacy: bool = False,
    sim_cutoff: float = 0.1,
    line_wash_threshold: float = 3.5,
) -> Type[DataChunk]:
    logger.add(f"{name}_analysis.log", rotation="1 days", colorize=True)
    output_path = Path(name)
    if not output_path.exists():
        output_path.mkdir()
    spectrum_path = Path(spectrum_path)
    if not spectrum_path.exists():
        raise FileNotFoundError("Spectrum file not found!")
    if spectrum_path.suffix == ".npy":
        load_func = np.load
    elif spectrum_path.suffix in [".txt", ".dat"]:
        load_func = np.loadtxt
    else:
        raise ValueError(f"File format not recognized: {spectrum_path.suffix}")
    data = load_func(spectrum_path).T
    # check that the shape of the loaded spectrum has rows as observations
    assert data.shape[-1] == 2
    logger.info(f"Number of elements: {data.size}")
    logger.info(f"Min/max frequency: {data[:,0].min():.4f}/{data[:,0].max():.4f}")
    if ".npz" in catalog_path:
        cat_type = "molsim"
    else:
        cat_type = "SPCAT"
    catalog = _load_catalog(catalog_path, type=cat_type)
    logger.info(f"There are {len(catalog.frequency)} catalog entries.")
    if not legacy:
        # process chunks of spectra, including GP noise estimation
        chunks, catalog_mask = extract_chunks(
            data, catalog, delta_v, vlsr, rbf_params, noise_params
        )
        # unroll the chunks, and reform them into a single DataChunk object
        frequency, intensity, noise, cat_indices = unroll_chunks(chunks)
        datachunk = DataChunk(
            frequency=frequency,
            intensity=intensity,
            catalog_index=cat_indices,
            noise=noise,
            mask=catalog_mask,
        )
    # use the version of the code from GOTHAM DR1
    else:
        datachunk = _legacy_filter_spectrum(
            catalog,
            data[:, 0],
            data[:, 1],
            vlsr,
            delta_v,
            block_interlopers,
            interloper_threshold,
            sim_cutoff,
            line_wash_threshold,
        )
    logger.info(f"Using {len(datachunk.catalog_index)} entries for analysis.")
    # dump stuff for later usage
    dump(catalog, output_path.joinpath("catalog.pkl"), compress=True)
    dump(
        datachunk.to_observation(observatory),
        output_path.joinpath("observation.pkl"),
        compress=True,
    )
    chunk_path = output_path.joinpath("datachunks.h5")
    if chunk_path.exists():
        os.remove(chunk_path)
    datachunk.to_hdf5(chunk_path)
    return datachunk
