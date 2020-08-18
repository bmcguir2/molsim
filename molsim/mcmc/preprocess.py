from typing import Type, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from itertools import repeat

import numpy as np
import h5py
from dask import array as da
from loguru import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from molsim.classes import Catalog
from molsim.mcmc import compute

from . import lineshapes


@dataclass
class DataChunk:
    frequency: np.ndarray
    intensity: np.ndarray
    catalog_index: int
    # technically we could instantiate with a NumPy array,
    # but None is probably faster
    noise: Any = None
    
    def to_hdf5(self, filename: str, **kwargs):
        with h5py.File(filename, **kwargs) as h5_file:
            h5_file.create_dataset("frequency", data=self.frequency)
            h5_file.create_dataset("intensity", data=self.intensity)
            h5_file.create_dataset("catalog_index", data=self.catalog_index)
            if self.noise is not None:
                h5_file.create_dataset("noise", data=self.noise)

    @classmethod
    def from_hdf5(cls, filename: str, **kwargs):
        h5_file = h5py.File(filename, **kwargs)
        frequency = da.from_array(h5_file["frequency"])
        intensity = da.from_array(h5_file["intensity"])
        catalog_index = da.from_array(h5_file["catalog_index"])
        if "noise" in h5_file:
            noise = da.from_array(h5_file["noise"])
        else:
            noise = None
        return cls(frequency, intensity, catalog_index, noise)


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


def extract_chunks(
    data: np.ndarray,
    catalog: Type[Catalog],
    delta_v: float = 5.0,
    rbf_params={},
    noise_params={},
    n_workers=4,
    verbose=False,
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
    # vectorized finding overlap regions in the spectrum
    mask = np.logical_and(min_freq <= catalog.frequency, catalog.frequency <= max_freq)
    # get indices to track which catalog entry is used
    indices = np.arange(catalog.frequency.size)[mask]
    # vectorized computation of the frequency offsets based on a doppler velocity
    offsets = compute.calculate_dopplerwidth_frequency(catalog.frequency[mask], delta_v)
    inband_freqs = np.vstack([indices, catalog.frequency[mask], offsets]).T
    logger.info(f"Expecting {len(inband_freqs)} chunks.")
    # parallelize the chunking and GP noise estimation; the latter does not have
    # native parallelization support, so we do a subprocess manually here
    # with ThreadPoolExecutor(n_workers) as executor:
    #     chunks = list(tqdm(executor.map(_compute_chunks, inband_freqs, [data] * len(inband_freqs)), total=len(inband_freqs)))
    chunks = list()
    for inband_data in tqdm(inband_freqs):
        try:
            chunks.append(
                _compute_chunks(inband_data, data, rbf_params, noise_params, verbose)
            )
        except ValueError:
            pass
    return chunks


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


def load_spectrum(
    spectrum_path: str,
    catalog: Type[Catalog],
    delta_v: float,
    rbf_params={},
    noise_params={},
    n_workers: int = 4,
    freq_range: Tuple[float, float] = (0.0, np.inf),
) -> Type[DataChunk]:
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
    chunks = extract_chunks(data, catalog, delta_v, rbf_params, noise_params, n_workers)
    frequency, intensity, noise, cat_indices = unroll_chunks(chunks)
    return DataChunk(
        frequency=frequency,
        intensity=intensity,
        catalog_index=cat_indices,
        noise=noise,
    )
