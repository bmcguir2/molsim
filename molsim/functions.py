import numpy as np
from molsim.constants import h, k
from molsim.classes import Spectrum
from molsim.utils import find_limits

def sum_spectra(sims,thin=True,Tex=None,Tbg=None,res=None,name='sum'):

	'''
	Adds all the spectra in the simulations list and returns a spectrum object.  By default,
	it assumes the emission is optically thin and will simply co-add the existing profiles.
	If thin is set to False, it will co-add and re-calculate based on the excitation/bg
	temperatures provided.  Currently, molsim can only handle single-excitation temperature
	co-adds with optically thick transmission, as it is not a full non-LTE radiative
	transfer program.  If a resolution is not specified, the highest resolution of the
	input datasets will be used.
	'''

		
	#first figure out what the resolution needs to be if it wasn't set
	if res is None:
		res = min([x.res for x in sims])
		
	#first we find out the limits of the total frequency coverage so we can make an
	#appropriate array to resample onto
	total_freq = np.concatenate([x.spectrum.freq_profile for x in sims])
	total_freq.sort()
	lls,uls = find_limits(total_freq,spacing_tolerance=2,padding=0)
		
	#now make a resampled array
	freq_arr = np.concatenate([np.arange(ll,ul,res) for ll,ul in zip(lls,uls)])	
	int_arr = np.zeros_like(freq_arr)	
	
	#make a spectrum to output
	sum_spectrum = Spectrum(name=name)
	sum_spectrum.freq_profile = freq_arr	

	if thin is True:		
		#loop through the stored simulations, resample them onto freq_arr, add them up
		for x in sims:
			int_arr0 = np.interp(freq_arr,x.spectrum.freq_profile,x.spectrum.int_profile,left=0.,right=0.)
			int_arr += int_arr0				
		sum_spectrum.int_profile = int_arr
		
	if thin is False:
		#if it's not gonna be thin, then we add up all the taus and apply the corrections
		for x in sims:
			int_arr0 = np.interp(freq_arr,x.spectrum.freq_profile,x.spectrum.tau_profile,left=0.,right=0.)
			int_arr += int_arr0
			
		#now we apply the corrections at the specified Tex
		J_T = ((h*freq_arr*10**6/k)*
			  (np.exp(((h*freq_arr*10**6)/
			  (k*Tex))) -1)**-1
			  )
		J_Tbg = ((h*freq_arr*10**6/k)*
			  (np.exp(((h*freq_arr*10**6)/
			  (k*Tbg))) -1)**-1
			  )			  
			
		int_arr = (J_T - J_Tbg)*(1 - np.exp(-int_arr))
		sum_spectrum.int_profile = int_arr

	return sum_spectrum
		