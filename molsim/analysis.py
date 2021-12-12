import numpy as np
from molsim.stats import get_rms
from molsim.constants import ckm
from molsim.utils import find_peaks, _get_res, find_nearest

def set_upper_limit(sim,obs,params={}):
	'''
	Automatically finds an upper limit for a simulation in an observation.
	
	params dictionary can contain:
		plot_name : str
			Name of the output plot, defaults to None (also plotting not implemented yet)
		vel_widths : float
			Number of FWHM on each side of line to calculate RMS values (defaults to 40.)
		tolerance : float
			How close to require the match between the best line and the rms to be (defaults to 0.01 or 1%)
		sigma : float
			What confidence level is desired on the upper limit (defaults to 1.0 sigma)
		return_result : bool
			Whether to return an upper limit results object that stores metadata and prints reports (defaults to False)
	'''
	
	#load in options from the params dictionary, and any defaults	
	plot_name = params['plot_name'] if 'plot_name' in params else None
	vel_widths = params['vel_widths'] if 'vel_widths' in params else 40.
	tolerance = params['tolerance'] if 'tolerance' in params else 0.01
	sigma = params['sigma'] if 'sigma' in params else 1.0
	return_result = params['return_result'] if 'return_result' in params else False
	
	#find the indices of the peaks in the simulation
	peak_indices = find_peaks(sim.spectrum.freq_profile,np.abs(sim.spectrum.int_profile),_get_res(sim.spectrum.freq_profile),sim.source.dV,is_sim=True)
	
	#get the frequencies and absolute values of the intensities in these regions
	peak_freqs = np.copy(sim.spectrum.freq_profile[peak_indices])
	peak_ints = np.copy(abs(sim.spectrum.int_profile[peak_indices]))
	
	#sort the arrays based on the intensity, and create some new ones to hold more info
	sort_idx = peak_ints.argsort()[::-1]
	peak_ints = peak_ints[sort_idx]
	peak_freqs = peak_freqs[sort_idx]
	peak_idx = peak_indices[sort_idx]
	peak_rms = np.copy(peak_ints)*0.
	peak_snr = np.copy(peak_ints)*0.
	
	#Go through and calculate RMS values, looking vel_widths on either side of the line for the RMS
	for i in range(len(peak_freqs)):
		ll_idx = find_nearest(obs.spectrum.frequency,peak_freqs[i] - vel_widths*sim.source.dV*peak_freqs[i]/ckm)
		ul_idx = find_nearest(obs.spectrum.frequency,peak_freqs[i] + vel_widths*sim.source.dV*peak_freqs[i]/ckm)
		rms = get_rms(obs.spectrum.Tb[ll_idx:ul_idx])
		#if the rms is NaN because there's no data in that region
		if np.isnan(rms) is True:
			peak_rms[i] = np.nan
			peak_snr[i] = 0.
		else:
			peak_rms[i] = rms
			peak_snr[i] = peak_ints[i]/rms
			
	#now find the maximum snr value and get the corresponding line frequency, rms, intensity, and index
	best_idx = np.argmax(peak_snr)
	best_freq = peak_freqs[best_idx]
	best_rms = sigma*peak_rms[best_idx]
	best_int = peak_ints[best_idx]
	
	#now continuously adjust the simulation column density until it matches the rms
	while abs(best_int - best_rms)/best_rms > tolerance:
		sim.source.column *= best_rms/best_int
		sim.update()
		best_int = np.nanmax(np.abs(sim.spectrum.int_profile[find_nearest(sim.spectrum.freq_profile,best_freq)]))
	
	if return_result is True:
		# Get the result class
		from molsim.classes import Ulim_Result
		# Make one
		result = Ulim_Result()
		# Start storing results
		result.line_frequency = best_freq
		result.line_intensity = best_int
		result.rms = best_rms/sigma
		result.sigma = sigma
		result.sim = sim
		result.obs = obs
	
		return result
	else:
		return