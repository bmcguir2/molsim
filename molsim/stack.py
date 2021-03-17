from dataclasses import dataclass
from typing import Tuple, List, Union, Type, Dict
from functools import lru_cache

from molsim.classes import Source, Simulation, Observation, Spectrum
from molsim.functions import sum_spectra, get_rms, find_peaks
from molsim.constants import ckm
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import h5py


class StackComponent(object):
    def __init__(self, source_size, dv, vlsr, tex, ncol):
        self._source_size = source_size
        self._dv = dv
        self._vlsr = vlsr
        self._tex = tex
        self._ncol = ncol

    @property
    def source_size(self) -> float:
        return self._source_size

    @property
    def dv(self) -> float:
        return self._dv

    @property
    def vlsr(self) -> float:
        return self._vlsr

    @property
    def tex(self) -> float:
        return self._tex

    @property
    def ncol(self) -> float:
        return self._ncol

    @property
    def source(self):
        return Source(
            size=self.source_size,
            dV=self.dv,
            velocity=self.vlsr,
            Tex=self.tex,
            column=self.ncol,
        )

    def make_simulation(self, molecule, ll, ul, obs, res: float = 0.0014):
        return Simulation(
            mol=molecule,
            ll=ll,
            ul=ul,
            observation=obs,
            source=self.source,
            line_profile="Gaussian",
            res=res,
        )


class AbstractStackSimulation(object):
    def __init__(self, molecule, ll, ul, obs, **kwargs):
        self._components = list()
        self._molecule = molecule
        self._ll = ll
        self._ul = ul
        self._obs = obs

    @property
    def components(self):
        return self._components

    @property
    def molecule(self):
        return self._molecule

    @property
    def ll(self):
        return self._ll

    @property
    def ul(self):
        return self._ul

    @property
    def obs(self):
        return self._obs

    @property
    def simulation(self):
        spectra = sum_spectra(
            [
                component.make_simulation(self.molecule, self.ll, self.ul, self.obs)
                for component in self.components
            ]
        )
        return spectra


class BenzonitrileStack(AbstractStackSimulation):
    def __init__(self, molecule, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        source_sizes, ncols, vlsrs = (
            [69.94, 97.97, 254.04, 259.53],
            [249259139101.892, 634917118945.907, 378570460131.327, 558873546309.917],
            [5.576, 5.765, 5.89, 6.02],
        )
        tex, dv = 8.59, 0.124
        for index in range(4):
            self._components.append(
                StackComponent(source_sizes[index], dv, vlsrs[index], tex, ncols[index])
            )


@dataclass
class SpectrumChunk:
    frequency: np.ndarray
    _intensity: np.ndarray
    center: np.ndarray
    _weight: float = 1.0
    _mask: np.ndarray = np.zeros(1, dtype=bool)

    def __post_init__(self):
        # set the default mask to take all values
        self._mask = np.ones_like(self.frequency, dtype=bool)

    def frequency_in_window(self, frequency: float) -> bool:
        """
        Checks whether a frequency is present in this chunk.

        Parameters
        ----------
        frequency : float
            Frequency to check for within the window.

        Returns
        -------
        bool
            True if the frequency is contained in the chunk.
        """
        return self.frequency.min() <= frequency <= self.frequency.max()

    def interpolate_intensity(self, new_velocity) -> np.ndarray:
        """
        Interpolates the weighted and masked intensity of this spectrum
        chunk onto a new velocity grid. Regions not covered by the original
        chunk will be set to `np.nan`.

        Parameters
        ----------
        new_velocity : [type]
            [description]

        Returns
        -------
        np.ndarray
            [description]
        """
        return np.interp(
            new_velocity,
            self.velocity,
            self.weighted_intensity,
            left=np.nan,
            right=np.nan,
        )

    def __len__(self) -> int:
        return len(self.frequency)

    def __repr__(self) -> str:
        return f"Elements: {len(self)}, center: {self.center:.4f} MHz, peak intensity: {self.peak_intensity:.4f}"

    def plot(self, weighted=False, velocity=False):
        """
        Make a `matplotlib` plot of this chunk. The arguments
        change what is plotted, either in velocity space, or
        whether to use the weighted intensity.

        Parameters
        ----------
        weighted : bool, optional
            Whether to plot the weighted intensity, by default False
        velocity : bool, optional
            Wheter to plot in velocity space, by default False
        """
        x, y = self.frequency, self.intensity
        if velocity:
            x = self.velocity
        if weighted:
            y = self.weighted_intensity
        return plt.plot(x, y)

    @property
    def intensity(self) -> np.ndarray:
        """
        Returns the masked intensity of this spectrum chunk.
        The `mask` property dictates which regions are to be
        set to `np.nan`, which are then excluded from calculations.

        Returns
        -------
        np.ndarray
            NumPy 1D array of masked intensities; regions outside
            of the region of interest are set to `np.nan`
        """
        masked_int = self._intensity.copy()
        masked_int[self.mask] = np.nan
        return masked_int

    @property
    def mask(self) -> np.ndarray:
        """
        Returns the current stored mask. If no default values
        are set, then no regions are protected, otherwise the
        mask is subsequently used when `SpectrumChunk.intensity`
        is requested; the mask sets regions that are `True` to
        `np.nan` to exclude from consideration.

        Returns
        -------
        np.ndarray
            NumPy 1D boolean mask corresponding to channels that
            will be set to `np.nan`
        """
        return self._mask

    @mask.setter
    def mask(self, parameters: Tuple[float]) -> None:
        """
        Sets the mask used for calculating intensities. The `parameters`
        argument is a five-tuple, containing the line width `dv`, the
        multiplier term `vel_roi` for determining how many channels to
        protect its intensity (`dv * vel_roi`), and the `rms_sigma` as
        the number of sigma away from the RMS to mask for intensity
        calculations, a `bias` value that can control the thresholding
        with intensity, and `freqs` as a NumPy 1D array of frequency
        values that need to be manually blocked.

        This is coded this way because setter methods can only take a
        single argument.

        For the intensity check, we actually use the larger of two
        numbers: either some multiple of the RMS, or a small number
        used for when the RMS is zero, which happens to be simulations.
        
        Finally, the `freq_mask` is added to the full mask because
        it corresponds to regions where we know there are definitely
        coincidences due to the same molecule. 
        The main use case here is using the knowledge of the simulation 
        to blank off regions that will have flux but aren't large enough
        to be picked off in intensity. For prolate tops, these are K-ladders
        that are too weak to be seen, but are actually still there.
        
        The final mask, i.e. the one that is set as an attribute, corresponds
        to regions (i.e. where the mask is True) that will be set to NaN.

        Parameters
        ----------
        parameters : Tuple[float]
            dv, vel_roi, rms_sigma, bias, and freqs
        """
        # unpack the arguments
        dv, vel_roi, rms_sigma, bias, freqs = parameters
        # isolate the ROI and work out the peak intensity. We will mask
        # everything else with NaN above this
        roi_mask = np.logical_and(
            -dv * vel_roi <= self.velocity, dv * vel_roi >= self.velocity
        )
        threshold = (get_rms(self._intensity) * rms_sigma) + bias
        # freqs is either None, or a NumPy 1D array of frequencies
        if freqs is not None:
            assert type(freqs) == np.ndarray
            # convert frequencies into equivalent velocity
            vels = (freqs - self.center) * ckm / self.center
            # mask regions corresponding to known interloping frequencies
            # from the same molecule
            freq_mask = np.sum([
                np.logical_and(
                    (-dv * vel_roi) + vel <= self.velocity, (dv * vel_roi) + vel >= self.velocity
                ) for vel in vels
            ], axis=0).astype(bool)
        else:
            freq_mask = np.zeros_like(roi_mask, dtype=bool)
        # combine the intensity and frequency masks
        blank_mask = (self._intensity >= threshold) + freq_mask
        # the ROI needs to be protected no matter what, so we set that
        # region to False always
        self._mask = blank_mask * (~roi_mask)

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float):
        if value < 0.0:
            raise ValueError("Weight can't be negative!")
        self._weight = value

    @property
    def peak_intensity(self) -> float:
        """
        Returns the maximum intensity, barring NaNs. This is
        useful for evaluating how much intensity is actually
        in the region of interest.

        Returns
        -------
        float
            Peak intensity excluding interloping flux.
        """
        return np.nanmax(self.intensity)

    @property
    def rms(self) -> float:
        """
        Calculates the RMS of the chunk, ignoring masked
        regions.

        Returns
        -------
        float
            RMS of the chunk, excluding masked regions.
        """
        return get_rms(self.intensity)

    @property
    def velocity(self) -> np.ndarray:
        """
        Returns the velocity at each spectral channel, using
        the set frequency center of this chunk.

        Returns
        -------
        np.ndarray
            NumPy 1D array containing velocities of each channel.
        """
        velocity = (self.frequency - self.center) * ckm / self.center
        return velocity

    @property
    def weighted_intensity(self) -> np.ndarray:
        """
        Returns the weighted, masked intensity of this chunk.

        Returns
        -------
        np.ndarray
            NumPy 1D array of weighted and masked intensities at
            each spectral channel.
        """
        return self.intensity * self.weight

    @property
    def snr(self) -> np.ndarray:
        """
        Returns the masked signal-to-noise ratio of this chunk.

        Returns
        -------
        np.ndarray
            Numpy 1D array of masked SNR at each channel.
        """
        return self.intensity / self.rms

    @property
    def weighted_snr(self) -> np.ndarray:
        """
        Returns the weighted and masked signal-to-noise ratio
        of each channel in this chunk.

        Returns
        -------
        np.ndarray
            NumPy 1D array of weighted and masked SNR at
            each channel.
        """
        return self.weighted_intensity / self.rms


class VelocityStack(object):
    """
    This class wraps the results of a velocity stack, basically
    providing an interface to the chunks that go into the stack,
    how much each contributes to the stack, and a quick way
    to visualize the results.
    """
    def __init__(
        self,
        velocity: np.ndarray,
        intensity: np.ndarray,
        sim_intensity: np.ndarray,
        obs_chunks: List[SpectrumChunk],
        sim_chunks: List[Spectrum],
    ) -> None:
        super().__init__()
        self._velocity = velocity
        self._intensity = intensity
        self._sim_intensity = sim_intensity
        self._obs_chunks = obs_chunks
        self._sim_chunks = sim_chunks

    def __repr__(self) -> str:
        return f"Number of chunks: {len(self._obs_chunks)}, is centered? {self.is_centered()}"

    def __len__(self) -> int:
        return len(self._obs_chunks)

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    @property
    def sim_intensity(self) -> np.ndarray:
        return self._sim_intensity

    @property
    def obs_chunks(self) -> List[SpectrumChunk]:
        return self._obs_chunks

    @property
    def sim_chunks(self) -> List[SpectrumChunk]:
        return self._sim_chunks

    @property
    @lru_cache(maxsize=None)
    def matched_filter(self) -> np.ndarray:
        """
        Return the matched filter spectrum by cross-correlation
        of the simulated and observed velocity stacks. The MF
        returned is in units of SNR.

        Returns
        -------
        np.ndarray
            Matched filter NumPy 1D array
        """
        matched_filter = np.correlate(self.intensity, self.sim_intensity, mode="same")
        matched_filter /= get_rms(matched_filter)
        return matched_filter

    @property
    @lru_cache(maxsize=None)
    def peak_mf_intensity(self) -> Tuple[float, float]:
        """
        Returns the peak matched filter intensity, and the
        corresponding velocity.

        Returns
        -------
        Tuple[float, float]
            Peak impulse response, and corresponding velocity
        """
        index = np.argmax(self.matched_filter)
        return (self.matched_filter[index], self.velocity[index])

    @property
    @lru_cache(maxsize=None)
    def contributions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the normalized contribution of each spectrum
        chunk, given as the sum of the weighted intensities
        of each chunk. Effectively, this is how much flux each
        spectrum chunk contributes to the stack.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two-tuple of NumPy 1D arrays: the first
            are the normalized contributions, and the
            second are indexes of contributions in ascending
            order.
        """
        obs_contributions = np.abs(
            [np.nansum(chunk.weighted_intensity) for chunk in self.obs_chunks]
        )
        obs_contributions /= obs_contributions.sum()
        # sorted in ascending order
        ordering = np.argsort(obs_contributions)[::-1]
        return (obs_contributions, ordering)

    @property
    @lru_cache(maxsize=None)
    def centers(self) -> np.ndarray:
        """
        Returns the frequency centers of each observational chunk.

        Returns
        -------
        np.ndarray
            NumPy 1D array of frequency centers
        """
        return np.array([chunk.center for chunk in self.obs_chunks])

    def plot(self):
        """
        Plot the matched filter spectrum, alongside the velocity
        stacks of the observation and simulation.

        Returns
        -------
        2-tuple of matplotlib figure and axis objects
        """
        fig, ax = plt.subplots()
        ax.plot(self.velocity, self.intensity, label="Obs.", alpha=0.7)
        ax.fill_between(self.velocity, self.sim_intensity, label="Sim.", alpha=0.6)
        ax.plot(self.velocity, self.matched_filter, label="Matched Filter")
        ax.legend()
        return (fig, ax)

    @lru_cache(maxsize=None)
    def is_centered(self, dv: float = 0.12, vel_roi: float = 10.0) -> bool:
        """
        Checks to see if the matched filter is centered by seeing
        if the peak impulse response lies within the velocity
        window of interest, specified by `vel_roi` number of line
        widths `dv`.

        Parameters
        ----------
        dv : float, optional
            Linewidth, by default 0.12
        vel_roi : float, optional
            Linewidth multiplier, by default 10.0

        Returns
        -------
        bool
            `True` if the matched filter is within the region
            of interest, otherwise `False`.
        """
        _, mf_vel = self.peak_mf_intensity
        return abs(mf_vel) <= dv * vel_roi

    def generate_report(self, dv: float = 0.12, vel_roi: float = 10.) -> Dict[str, str]:
        mf_peak, peak_vel = self.peak_mf_intensity
        baseline = get_rms(self.matched_filter)
        centered = self.is_centered(dv, vel_roi)
        data = {
            "peak_response": f"{mf_peak:.3f}",
            "mf_baseline": f"{baseline:.5f}",
            "is_centered": bool(centered)
        }
        return data

    def flux_check(self, num_points: Union[int, None] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Perform a linear fit to the simulated and observed peak intensities
        in log-log space. If there are weird things happening in this plot
        there are likely interlopers.

        Parameters
        ----------
        num_points : Union[int, None], optional
            Number of points to use for interpolation after fitting, by default None.
            If `None`, we will use two times the number of data points.

        Returns
        -------
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarrary]]
            Nested two-tuple; the first pair are x, y (sim/obs)
            data points, and the second pair are the corresponding
            values from the linear fit.
        """
        x = np.log10([chunk.peak_intensity for chunk in self.sim_chunks])
        y = np.log10([chunk.peak_intensity for chunk in self.obs_chunks])
        # do a least squares fit to the log-log data
        fit = np.polynomial.Polynomial.fit(x, y, deg=1)
        if not num_points:
            num_points = len(x) * 2
        model_x, pred_y = fit.linspace(num_points)
        return ((x, y), (model_x, pred_y))


def generate_spectrum_chunks(
    frequency: np.ndarray,
    intensity: np.ndarray,
    centers: np.ndarray,
    vel_width: float = 40.0,
    n_workers: int = 1,
) -> List[SpectrumChunk]:
    """
    Function to generate `SpectrumChunk` objects, given an `Observation`, an
    array of frequency centers, and the width in velocity to isolate for each
    chunk. The `centers` array is expected to already be shifted to the
    source frame, i.e. the frequency shift expected in the `Observation`.
    Otherwise, you'll be stacking nothing but noise :)

    Parameters
    ----------
    observation : Observation
        Instance of an `molsim` `Observation` object. This looks for
        the `spectrum.frequency` and `spectrum.Tb` attributes for the
        frequency and intensities of each chunk.
    centers : np.ndarray
        NumPy 1D array containing frequency centers to use for isolation.
        _These centers are assumed to be shifted to the source frame!_
    vel_width : float, optional
        Velocity value to use for defining a chunk, by default 40.
        The chunk will contain data corresponding to +/-vel_width.
    n_workers : int, optional
        Number of threads to use for parallelization. By default, 1
        which uses a single thread, but still wrapped in the `joblib`
        parallel context.

    Returns
    -------
    List[SpectrumChunk]
        List of `SpectrumChunk` objects.
    """
    # calculate window widths in velocity space
    window_widths = vel_width * centers / ckm
    chunks = list()
    # loop over each catalog frequency, and try to extract out windows
    with Parallel(n_jobs=n_workers, prefer="threads", require="sharedmem") as parallel:
        chunks = parallel(
            delayed(_make_chunk)(frequency, intensity, center, width)
            for center, width in zip(centers, window_widths)
        )
    # remove chunks without enough data
    chunks = list(filter(lambda x: x is not None, chunks))
    return chunks


def _make_chunk(
    frequency: np.ndarray, intensity: np.ndarray, center: float, width: float
) -> Union[None, Type[SpectrumChunk]]:
    """
    Quasi private function that generates a `SpectrumChunk` after calculating
    the approriate frequency mask. This was mainly encapsulated in this function
    to help the parallelization, and so the user is not typically expected
    to call this function directly.

    Parameters
    ----------
    frequency : np.ndarray
        NumPy 1D array containing frequencies
    intensity : np.ndarray
        NumPy 1D array containing intensities
    center : float
        Frequency center used to extract a chunk
    width : float
        Window width used to extract a chunk

    Returns
    -------
    Union[None, Type[SpectrumChunk]]
        Returns a SpectrumChunk if there are more than two elements
        in the array after masking, otherwise None
    """
    lower, upper = center - width, center + width
    mask = np.logical_and(frequency >= lower, frequency <= upper)
    if mask.sum() > 2:
        # make a contiguous copy of the arrays so we don't modify them
        freq_array, int_array = frequency[mask].copy(), intensity[mask].copy()
        return SpectrumChunk(freq_array, int_array, center)
    else:
        return None


def velocity_stack(
    chunks: List[SpectrumChunk],
    vel_width: float = 40.0,
    resolution: float = 0.0014,
    rms_weights: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the velocity stack, given a list of `SpectrumChunk` objects, and
    the parameters used to define the velocity grid that all the calculations
    will be performed on.

    This function works by gathering up all the interpolated intensities, which
    are also weighted and masked according to their regions of interest (see
    the `SpectrumChunk` object for more details); i.e. we assume the weights
    have been set correctly already.

    This function is quite atomic, and so chances are you are looking for the
    higher level function, `velocity_stack_pipeline`.

    Parameters
    ----------
    chunks : List[SpectrumChunk]
        List of `SpectrumChunk` objects
    vel_width : float, optional
        Sets the region of velocity we care about, by default 40.0.
        This corresponds to +/-vel_width as the limits.
    resolution : float, optional
        Step size of the grid in velocity, by default 0.0014.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [description]
    """
    new_velocity = np.arange(-vel_width, vel_width, resolution)
    # gather up the weighted and masked intensities of each chunk,
    # interpolated to the same velocity grid
    intensities = np.vstack(
        [chunk.interpolate_intensity(new_velocity) for chunk in chunks]
    )
    # Work out how many of the channels in each chunk are actually
    # real signal so we can weight properly
    if rms_weights is None:
        rms_counts = (~np.isnan(intensities)).astype(float)
        rms_values = np.array([chunk.rms for chunk in chunks])[:, None]
        rms_counts *= rms_values ** 2.0
        # rms weighting on a channel basis
        rms_weights = rms_counts.sum(axis=0)
    # do the stack by summing up along the velocity axis
    stacked_int = np.nansum(intensities, axis=0) / rms_weights
    # set regions that were NaN'd to zero now that we're done
    stacked_int[np.isnan(stacked_int)] = 0.0
    return (new_velocity, stacked_int, rms_weights)


def velocity_stack_pipeline(
    spectrum: Spectrum,
    observation: Observation,
    vel_width: float = 40.0,
    resolution: float = 0.0014,
    dv: float = 0.12,
    vel_roi: float = 10.0,
    rms_sigma: float = 3.0,
    n_workers: int = 1,
) -> Type[VelocityStack]:
    """
    High-level function for performing a velocity stack. Takes `Simulation` object,
    which provides information how much flux to expect per transition, as well as
    determines which frequency centers to chunk and stack.

    The chunk generation is parallelized using `joblib` with threading and shared
    memory. This step is the most time exhaustive so far, as it requires looping
    over the detected peaks, which can be hundreds.

    Parameters
    ----------
    spectrum : Spectrum
        Instance of a `molsim.Spectrum` object, which
        is produced from a `Simulation`
    observation : Observation
        Instance of a `molsim.Observation` object
    vel_width : float, optional
        Window size in velocity, by default +/-40.
    resolution : float, optional
        Resolution of the window, by default 0.0014 km/s
    dv : float, optional
        Nominal line width, by default 0.12
    vel_roi : float, optional
        Number of line widths to define the region of interest,
        by default 10.
    rms_sigma : float, optional
        Multiples of RMS to use as a threshold for interloper
        masking, by default 3.

    Returns
    -------
    VelocityStack
        Instance of the `VelocityStack` class, which
        wraps the results
    """
    obs_x, obs_y = observation.spectrum.frequency, observation.spectrum.Tb
    sim_x, sim_y = spectrum.freq_profile, spectrum.int_profile
    # find the peaks in the simulated spectrum to use as the frequency
    # center. This is used because it's more resilient to weird lineshapes.
    peak_indices = find_peaks(
        sim_x, sim_y, resolution, min_sep=vel_roi * dv, is_sim=True, sigma=rms_sigma
    )
    centers = sim_x[peak_indices]
    obs_chunks = generate_spectrum_chunks(obs_x, obs_y, centers, vel_width, n_workers)
    sim_chunks = generate_spectrum_chunks(sim_x, sim_y, centers, vel_width, n_workers)
    # for each chunk, set the velocity mask to protect the intensity of each ROI
    for chunks in zip(obs_chunks, sim_chunks):
        for chunk_type, chunk in enumerate(chunks):
            if chunk_type == 0:
                bias = 0.
                # find which windows we should definitely mask because
                # we know there's something there
                coin_mask = np.asarray([chunk.frequency_in_window(freq) for freq in centers])
                coincidences = centers[coin_mask]
                # sometimes we don't have coincidences and that's okay
                if coincidences.sum() < 1:
                    coincidences = None
            else:
                # this shifts the threshold for flux masking; for simulations
                # we impose a large negative offset to zero everything out
                bias = -10.
                coincidences = None
            chunk.mask = (dv, vel_roi, rms_sigma, bias, coincidences)
    # the simulated data is used to weight the stacking
    expected_intensities = sim_y[peak_indices]
    max_expected = expected_intensities.max()
    # for each chunk, we set the weight to be equal to the contribution of
    # this specific transition compared to the other transitions, divided
    # by the observational RMS squared
    for obs_chunk, sim_chunk in zip(obs_chunks, sim_chunks):
        obs_rms = obs_chunk.rms
        cum_sim = np.nansum(sim_chunk.intensity)
        for chunk in [obs_chunk, sim_chunk]:
            chunk.weight = (cum_sim / max_expected) / (obs_rms ** 2.0)
    # perform the velocity stack for both observation and simulation
    (obs_stack_x, obs_stack_y, rms_weights) = velocity_stack(
        obs_chunks, vel_width, resolution
    )
    (_, sim_stack_y, _) = velocity_stack(sim_chunks, vel_width, resolution, rms_weights)
    # calculate the RMS of the observational velocity stack
    stack_rms = get_rms(obs_stack_y)
    obs_stack_y /= stack_rms
    sim_stack_y /= stack_rms
    result = VelocityStack(
        obs_stack_x, obs_stack_y, sim_stack_y, obs_chunks, sim_chunks
    )
    return result