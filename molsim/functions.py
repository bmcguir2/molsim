import numpy as np
from molsim.constants import h, k, ckm
from molsim.classes import Spectrum, Continuum
from molsim.utils import find_limits, _get_res, _find_nans, find_peaks, find_nearest, _find_ones
from molsim.stats import get_rms
from molsim.file_handling import load_mol
from datetime import date
import matplotlib.pyplot as plt
import matplotlib


def sum_spectra(sims,thin=True,Tex=None,Tbg=None,res=None,noise=None,override_freqs=None,planck=False,name='sum',tau_threshold=1000.,spacing_tolerance=None):

	'''
	Adds all the spectra in the simulations list and returns a spectrum object.  By default,
	it assumes the emission is optically thin and will simply co-add the existing profiles.
	If thin is set to False, it will co-add and re-calculate based on the excitation/bg
	temperatures provided.  Currently, molsim can only handle single-excitation temperature
	co-adds with optically thick transmission, as it is not a full non-LTE radiative
	transfer program.  If a resolution is not specified, the highest resolution of the
	input datasets will be used.
	
	If the user wants back the summed spectra on an exact set of frequencies, they can specify
	that array (a numpy array) as override_freqs.
	
	
	'''

	if any([x.use_obs for x in sims]):
		if res is None and override_freqs is None:
			raise RuntimeError("Resolution cannot be determined since one or more spectra enabled `use_obs`. Please specify `res` or `override_freqs` to prevent this error.")
		
	#first figure out what the resolution needs to be if it wasn't set
	if res is None:
		res = min([x.res for x in sims])

	#then figure out the spacing tolerance needed if it wasn't set
	if spacing_tolerance is None:
		# 4 in the spacing_tolerance is a magic number that can deal with doubly-repeated frequencies per simulated spectrum
		spacing_tolerance = 4*len(sims)
		
	
	#check if override_freqs has been specified, and if so, use that.
	if override_freqs is not None:
		freq_arr = override_freqs
		
	else:	
		#first we find out the limits of the total frequency coverage so we can make an
		#appropriate array to resample onto
		total_freq = np.concatenate([x.spectrum.freq_profile for x in sims])
		#eliminate all duplicate entries
		total_freq = np.array(list(set(total_freq)))
		total_freq.sort()
		lls,uls = find_limits(total_freq,spacing_tolerance=spacing_tolerance,padding=0)
		
		#now make a resampled array
		freq_arr = np.concatenate([np.arange(ll,ul,res) for ll,ul in zip(lls,uls)])	
	
	#now make an identical array to hold intensities
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
	
		#Check to see if the user has specified a Tbg
		if Tbg is None:
			print('If summing for the optically thick condition, either a constant Tbg or an appropriate Continuum object must be provided.  Operation aborted.')
			return
		#Otherwise if we have a continuum object, we use that to calculate the Tbg at each point in freq_arr generated above
		if isinstance(Tbg, Continuum):
			sum_Tbg = Tbg.Tbg(freq_arr)
		else:
			sum_Tbg = Tbg

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
			  (k*sum_Tbg))) -1)**-1
			  )			  
		
		int_arr = (J_T - J_Tbg)*(1 - np.exp(-int_arr))

		##########################
		# For Spectra in Jy/Beam #
		##########################

		if planck is True:
		
			#Collect the ranges over which different beam sizes are in play.
			omegas = [] #solid angle
			omegas_lls = [] #lower limits of frequency ranges covered by a solid angle
			omegas_uls = [] #upper limits of frequency ranges covered by a solid angle
			
			#loop through the simulations and extract the lls, uls, and omegas (from the synthesized beams)
			for sim in sims:
				for ll in sim.ll:
					omegas_lls.append(ll)
					omegas.append(sim.observation.observatory.synth_beam[0]*sim.observation.observatory.synth_beam[1])
				for ul in sim.ul:
					omegas_uls.append(ul)
				
			#now we need an array that is identical to freq_arr, but holds the omega values at each point
			omega_arr = np.zeros_like(freq_arr)
			
			#and now we have to fill it, given the ranges we have data for.
			#we start by making arrays to hold all possible omega values
			tmp_omegas = []
			for omega,ll,ul in zip(omegas,omegas_lls,omegas_uls):
				tmp_omega = np.zeros_like(freq_arr)
				tmp_omega[np.where(np.logical_and(freq_arr >= ll, freq_arr <= ul))] = omega
				tmp_omegas.append(np.array(tmp_omega))  
	
			#now go through and flatten it into a single array, keeping the biggest omega value at each frequency
			#this is arbitrary. It's not possible to sum spectra at a point with more than one omega value.  The user has to make
			#sure they aren't doing this.  We keep just the largest.
			omega_arr = np.maximum.reduce(tmp_omegas)
			
			#now we can do the actual conversion to Planck scale Jy/beam.  We can only operate on non-zero values.
			mask = np.where(int_arr != 0)[0]
			int_arr[mask] = (3.92E-8 * (freq_arr[mask]*1E-3)**3 *omega_arr[mask] / (np.exp(0.048*freq_arr[mask]*1E-3/int_arr[mask]) - 1))
					
		sum_spectrum.int_profile = int_arr
			

	#add in noise, if requested
	if noise is not None:
		#initiate the random number generator
		rng = np.random.default_rng()
		
		#generate a noise array the same length as the simulation,
		noise_arr = rng.normal(0,noise,len(sum_spectrum.int_profile))
		
		#add it in
		sum_spectrum.int_profile += noise_arr

	
	return sum_spectrum
	
def resample_obs(x_arr,y_arr,res,return_spectrum=False):
	'''
	Resamples x_arr and y_arr to a resolution of 'res' in whatever units x_arr is and returns
	them as numpy arrays if return_spectrum is False or as a Spectrum object if return_spectrum
	is True.
	'''	
	
	lls,uls = find_limits(x_arr)
	new_x = np.array([])
	
	for x,y in zip(lls,uls):
		new_x = np.concatenate((new_x,np.arange(x,y,res)))
		
	new_y = np.interp(new_x,x_arr,y_arr,left=np.nan,right=np.nan)
	
	if return_spectrum is False:
		return new_x,new_y
	else:
		return Spectrum(frequency=new_x,Tb=new_y)	

def velocity_stack(params):

	'''
	Perform a velocity stack.  Requires a params catalog for all the various options.
	Here they are, noted as required, or otherwise have defaults:
	
	name: a name for this spectrum object. String.  Default: 'stack'
	selection : 'peaks' or 'lines'. Default: 'lines'
	freq_arr : the array of frequencies. Required
	int_arr : the array of intensities. Required
	freq_sim: the array of simulated frequencies.  Required
	int_sim: the array of simulated intensities. Required
	res_inp : resolution of input data [MHz].  Calculates if not given
	dV : FWHM of lines [km/s]. Required.
	dV_ext : How many dV to integrate over.  Required if 'lines' selected.
	vlsr: vlsr [km/s]. Default: 0.0 
	vel_width : how many km/s of spectra on either side of a line to stack [km/s].  Required.
	v_res: desired velocity resolution [km/s].  Default: 0.1*dV
	drops: id's of any chunks to exclude.  List.  Default: []
	blank_lines : True or False.  Default: False
	blank_keep_range: range over which not to blank lines.  List [a,b].  Default: 3*dV
	flag_lines: True or False. Default: False
	flag_sigma : number of sigma over which to consider a line an interloper.  Float.  Default: 5.
	n_strongest: stack the strongest x lines.  Integer.  Default: All lines.
	n_snr: stack the x highest snr lines.  Integer.  Default: All lines.
	return_snr: output arrays of the snrs stacked plus the snr of the stack itself. True or False. Default: False 
	'''
	
	#define an obs_chunk class to hold chunks of data to stack
	
	class ObsChunk(object):

		def __init__(self,freq_obs,int_obs,freq_sim,int_sim,peak_int,id,cfreq):
	
			self.freq_obs = freq_obs #frequency array to be stacked
			self.int_obs = int_obs #intensity array to be stacked
			self.freq_sim = freq_sim #simulated frequency array to be stacked
			self.int_sim = int_sim #simulated intensity array to be stacked
			self.peak_int = peak_int #peak intensity for this chunk
			self.id = id #id of this chunk
			self.cfreq = cfreq #center frequency of the chunk
			self.flag = False #flagged as not to be used
			self.rms = None #rms of the chunk
			self.velocity = None #to hold the velocity array
			self.test = False
			
			self.check_data()
			if self.flag is False:
				self.set_rms()
				self.set_velocity()
				self.set_sim_velocity()
			
			return
			
		def check_data(self):
			#check if we have enough data here or if we ended up near an edge or a bunch of nans
			if len(self.freq_obs) < 2:
				self.flag = True
				return
			#check if we have more nans than not, and if so, skip it
			if np.count_nonzero(~np.isnan(self.int_obs)) < np.count_nonzero(np.isnan(self.int_obs)):
				self.flag = True
				return
			#check if peak_int is 0.0, in which case skip it
			if self.peak_int == 0:
				self.flag = True
				return
			return
			
		def set_rms(self):
			self.rms = get_rms(self.int_obs)
			return	
			
		def set_velocity(self):
			vel = np.zeros_like(self.freq_obs)
			vel += (self.freq_obs - self.cfreq)*ckm/self.cfreq
			self.velocity = vel
			return	
			
		def set_sim_velocity(self):
			sim_vel = np.zeros_like(self.freq_sim)
			sim_vel += (self.freq_sim - self.cfreq)*ckm/self.cfreq
			self.sim_velocity = sim_vel
			return				


	#unpacking the dictionary into local variables for ease of use
	options = params.keys()
	name = params['name'] if 'name' in options else 'stack'
	freq_arr = np.copy(params['freq_arr'])
	int_arr = np.copy(params['int_arr'])
	freq_sim = np.copy(params['freq_sim'])
	int_sim = np.copy(params['int_sim'])
	res_inp = params['res_inp'] if 'res_inp' in options else _get_res(freq_arr)
	dV = params['dV']
	dV_ext = params['dV_ext'] if 'dV_ext' in options else None
	vlsr = params['vlsr'] if 'vlsr' in options else 0.0
	vel_width = params['vel_width']
	v_res = params['v_res'] if 'v_res' in options else 0.1*dV
	drops = params['drops'] if 'drops' in options else []
	blank_lines = params['blank_lines'] if 'blank_lines' in options else False
	blank_keep_range = params['blank_keep_range'] if 'blank_keep_range' in options else [-3*dV,3*dV]
	flag_lines = params['flag_lines'] if 'flag_lines' in options else False
	flag_sigma = params['flag_sigma'] if 'flag_sigma' in options else 5.	
	n_strongest = params['n_strongest'] if 'n_strongest' in options else None
	n_snr = params['n_snr'] if 'n_snr' in options else None
	return_snr = params['return_snr'] if 'return_snr' in options else False

	#initialize a spectrum object to hold the stack and name it
	stacked_spectrum = Spectrum(name=name)
	
	#determine the locations to stack and their intensities, either with peaks or lines
	if params['selection'] == 'peaks':
		peak_indices = find_peaks(freq_sim,int_sim,res_inp,dV,is_sim=True)
		peak_freqs = freq_sim[peak_indices]
		peak_ints = int_sim[peak_indices]
		
	if params['selection'] == 'lines':
		peak_indices = find_peaks(freq_sim,int_sim,res_inp,dV*dV_ext,is_sim=True)	
		peak_freqs = freq_sim[peak_indices]
		freq_widths = dV*dV_ext*peak_freqs/ckm
		lls = np.asarray([find_nearest(freq_sim,(x-y/2)) for x,y in zip(peak_freqs,freq_widths)])
		uls = np.asarray([find_nearest(freq_sim,(x+y/2)) for x,y in zip(peak_freqs,freq_widths)])
		peak_ints = np.asarray([np.nansum(int_sim[x:y]) for x,y in zip(lls,uls)])
		
	#choose the n strongest lines, if that is specified
	if n_strongest is not None:
		sort_idx = np.flip(np.argsort(peak_ints))
		if n_strongest > len(peak_ints):
			pass
		else:
			peak_ints = peak_ints[sort_idx][:n_strongest]	
			peak_freqs = peak_freqs[sort_idx][:n_strongest]
			
	#choose the n highest snr lines, if that is instead specified
	if n_snr is not None:
		if n_snr > len(peak_ints):
			pass
		else:		
			freq_widths = vel_width*peak_freqs/ckm
			lls_obs = np.asarray([find_nearest(freq_arr,x-y) for x,y in zip(peak_freqs,freq_widths)])
			uls_obs = np.asarray([find_nearest(freq_arr,x+y) for x,y in zip(peak_freqs,freq_widths)])		
			line_noise = np.asarray([get_rms(int_arr[x:y]) for x,y in zip(lls_obs,uls_obs)])
			line_snr = peak_ints/line_noise
			sort_idx = np.flip(np.argsort(line_snr))
			peak_ints = peak_ints[sort_idx][:n_snr]	
			peak_freqs = peak_freqs[sort_idx][:n_snr]
	
	
	#split out the data to use, first finding the appropriate indices for the width range we want
	freq_widths = vel_width*peak_freqs/ckm
	lls_obs = np.asarray([find_nearest(freq_arr,x-y) for x,y in zip(peak_freqs,freq_widths)])
	uls_obs = np.asarray([find_nearest(freq_arr,x+y) for x,y in zip(peak_freqs,freq_widths)])
	lls_sim = np.asarray([find_nearest(freq_sim,x-y) for x,y in zip(peak_freqs,freq_widths)])
	uls_sim = np.asarray([find_nearest(freq_sim,x+y) for x,y in zip(peak_freqs,freq_widths)])						
		
	obs_chunks = [ObsChunk(np.copy(freq_arr[x:y]),np.copy(int_arr[x:y]),np.copy(freq_sim[a:b]),np.copy(int_sim[a:b]),peak_int,c,d) for x,y,a,b,peak_int,c,d in zip(lls_obs,uls_obs,lls_sim,uls_sim,peak_ints,range(len(uls_sim)),peak_freqs)]

	#flagging
	for obs in obs_chunks:
		#already flagged, move on
		if obs.flag is True:
			continue
		#make sure there's data at all.
		if len(obs.freq_obs) == 0:
			obs.flag = True
			continue	
		#drop anything in drops
		if obs.id in drops:
			obs.flag = True
			continue	
		#blank out lines not in the center to be stacked
		if blank_lines is True:			
			#Find the indices corresponding to the safe range
			ll_obs = find_nearest(obs.freq_obs,obs.cfreq - blank_keep_range[1]*obs.cfreq/ckm)
			ul_obs = find_nearest(obs.freq_obs,obs.cfreq - blank_keep_range[0]*obs.cfreq/ckm)
			mask = np.concatenate((np.where(abs(obs.int_obs[:ll_obs]) > flag_sigma * obs.rms)[0],np.where(abs(obs.int_obs[ul_obs:]) > flag_sigma * obs.rms)[0]+ul_obs))
			obs.int_obs[mask] = np.nan
			obs.set_rms()
			obs_nans_lls,obs_nans_uls = _find_nans(obs.int_obs)
			obs_nans_freqs_lls = obs.int_obs[obs_nans_lls]
			obs_nans_freqs_uls = obs.int_obs[obs_nans_uls]
			sim_nans_lls = [find_nearest(obs.int_sim,x) for x in obs_nans_freqs_lls]
			sim_nans_uls = [find_nearest(obs.int_sim,x) for x in obs_nans_freqs_uls]
			for x,y in zip(sim_nans_lls,sim_nans_uls):
				obs.int_sim[x:y] = np.nan			
				
		#if we're flagging lines in the center, do that now too
		if flag_lines is True:
			if np.nanmax(obs.int_obs) > flag_sigma*obs.rms:
				obs.flag = True
				continue
				
	#setting and applying the weights
	max_int = max(peak_ints)
	for obs in obs_chunks:
		if obs.flag is False:
			obs.weight = obs.peak_int/max_int
			obs.weight /= obs.rms**2
			obs.int_weighted = obs.int_obs * obs.weight
			obs.int_sim_weighted = obs.int_sim * obs.weight	
			
			
	#Generate a velocity array to interpolate everything onto				
	velocity_avg = np.arange(-vel_width,vel_width,v_res)	
	
	#go through all the chunks and resample them, setting anything that is outside the range we asked for to be nans.
	for obs in obs_chunks:
		if obs.flag is False:
			obs.int_samp = np.interp(velocity_avg,obs.velocity,obs.int_weighted,left=np.nan,right=np.nan)
			obs.int_sim_samp = np.interp(velocity_avg,obs.sim_velocity,obs.int_sim_weighted,left=np.nan,right=np.nan)		
	
	#Now we loop through all the chunks and add them to a list, then convert to an numpy array.  We have to do the same thing w/ RMS values to allow for proper division.
	interped_ints = []
	interped_rms = []
	interped_sim_ints = []
	
	for obs in obs_chunks:
		if obs.flag is False:
			interped_ints.append(obs.int_samp)
			interped_rms.append(obs.rms)
			interped_sim_ints.append(obs.int_sim_samp)
	
	interped_ints = np.asarray(interped_ints)
	interped_rms = np.asarray(interped_rms)
	interped_sim_ints = np.asarray(interped_sim_ints)
	
	#we're going to now need a point by point rms array, so that when we average up and ignore nans, we don't divide by extra values.
	rms_arr = []
	for x in range(len(velocity_avg)):
		rms_sum = 0
		for y in range(len(interped_rms)):
			if np.isnan(interped_ints[y][x]):
				continue
			else:
				rms_sum += interped_rms[y]**2
		rms_arr.append(rms_sum)
	rms_arr	= np.asarray(rms_arr)
	rms_arr[rms_arr==0] = np.nan
	
	#add up the interped intensities, then divide that by the rms_array
	int_avg = np.nansum(interped_ints,axis=0)/rms_arr
	int_sim_avg = np.nansum(interped_sim_ints,axis=0)/rms_arr
	
	#drop some edge channels
	int_avg = int_avg[5:-5]
	int_sim_avg = int_sim_avg[5:-5]
	velocity_avg = velocity_avg[5:-5]
	
	#Get the final rms, and divide out to get to snr.
	rms_tmp = get_rms(int_avg)
	int_avg /= rms_tmp
	int_sim_avg /= rms_tmp
	
	#store everything in the spectrum object and return it
	stacked_spectrum.velocity = np.copy(velocity_avg)
	stacked_spectrum.snr = np.copy(int_avg)
	stacked_spectrum.int_sim = np.copy(int_sim_avg)
						
	if return_snr is False:
		return stacked_spectrum
	if return_snr is True:	
		ll = find_nearest(velocity_avg,-dV*dV_ext)
		ul = find_nearest(velocity_avg,dV*dV_ext)
		stack_int = np.nansum(int_avg[ll:ul])
		stack_rms = get_rms(int_avg[ll:ul])
		stack_snr = stack_int*1E5
		return stacked_spectrum,line_snr[sort_idx][:n_snr],stack_snr
	
def matched_filter(data_x,data_y,filter_y,name='mf'):
	'''
	Perform a matched filter analysis on data_x,data_y using filter_y
	'''
	
	#do the filter and normalization to SNR scale
	mf_y = np.correlate(data_y,filter_y,mode='valid')
	mf_y /= get_rms(mf_y)
	
	#trim off the edges of the velocity data to match the range of the filter response
	nchans = round(len(mf_y)/2)
	c_chan = round(len(data_x)/2)
	mf_x = np.copy(data_x[c_chan-nchans:c_chan+nchans])
	#make sure there's no rounding errors
	if abs(len(mf_x) - len(mf_y)) == 1:
		if len(mf_x) > len(mf_y):
			mf_x = mf_x[:-1]
		else:
			mf_y = mf_y[:-1]	
	
	#load the result into a Spectrum object and return it.
	mf = Spectrum(name=name)
	mf.velocity = np.copy(mf_x)
	mf.snr = np.copy(mf_y)
	
	return mf
	
def convert_spcat(filein,params={}):
	'''
	Converts an SPCAT catalog to a molsim catalog and updates the metadata as indicated
	in params.
	'''
	
	settings = {'fileout'			:	filein[:-1-len(filein.split('.')[-1])],
				'version'			:	1.0,
				'source'			:	None,
				'last_update'		:	date.today().strftime("%B %d, %Y"),
				'contributor_name'	:	None,
				'contributor_email'	:	None,
				'notes'				:	None,
				'refs'				:	None,
				}
				
	for x in params:
		if x in settings:
			settings[x] = params[x]
			
	mol = load_mol(filein, type='SPCAT')
	
	mol.catalog.version = settings['version']
	mol.catalog.source = settings['source']
	mol.catalog.last_update = settings['last_update']
	mol.catalog.contributor_name = settings['contributor_name']
	mol.catalog.contributor_email = settings['contributor_email']
	mol.catalog.notes = settings['notes']
	mol.catalog.refs = settings['refs']
	
	mol.catalog.export_cat(settings['fileout'])			
	
	return