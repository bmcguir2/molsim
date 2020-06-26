import numpy as np
from numba import njit
from molsim.constants import ccm, cm, ckm, h, k, kcm
import math
import warnings
from scipy import stats, signal

def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx 

def _trim_arr(arr,lls,uls,key_arr=None,return_idxs=False,ll_idxs=None,ul_idxs=None):
	'''
	Trims the input array to the limits specified.  Optionally, will get indices from 
	the key_arr for trimming instead.
	'''
	
	if ll_idxs is not None:
		return np.concatenate([arr[ll_idx:ul_idx+1] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)])
	
	if key_arr is None:
		ll_idxs = [find_nearest(arr,x) for x in lls]
		ul_idxs = [find_nearest(arr,x) for x in uls]
	else:
		ll_idxs = [find_nearest(key_arr,x) for x in lls]
		ul_idxs = [find_nearest(key_arr,x) for x in uls]

	if return_idxs is False:
		return np.concatenate([arr[ll_idx:ul_idx+1] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)])
	else:
		return np.concatenate([arr[ll_idx:ul_idx+1] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)]),ll_idxs,ul_idxs
		
@njit
def _make_gauss(freq0,int0,freq,dV,ckm):
	return int0*np.exp(-((freq-freq0)**2/(2*((dV*freq0/ckm)/2.35482)**2)))

def _make_level_dict(qn1low,qn2low,qn3low,qn4low,qn5low,qn6low,qn7low,qn8low,qn1up,qn2up,
					qn3up,qn4up,qn5up,qn6up,qn7up,qn8up,frequency,elow,gup,
					qn_list_low,qn_list_up,level_qns,level_dict,qnstrfmt=None):

	#a list to hold levels
	levels = []
	
	#we need to sort out unique levels from our catalog.  Those will have unique quantum
	#numbers. When we find a match to a lower level, add the info in.
	for x in range(len(frequency)):
		qnstr_low = qn_list_low[x]
		level_dict[qnstr_low] = {'energy'	:	elow[x],
								 'g'		:	None,
								 'g_flag'	:	False,
								 'qn1'		:	qn1low[x] if qn1low is not None else None,
								 'qn2'		:	qn2low[x] if qn2low is not None else None,
								 'qn3'		:	qn3low[x] if qn3low is not None else None,
								 'qn4'		:	qn4low[x] if qn4low is not None else None,
								 'qn5'		:	qn5low[x] if qn5low is not None else None,
								 'qn6'		:	qn6low[x] if qn6low is not None else None,
								 'qn7'		:	qn7low[x] if qn7low is not None else None,
								 'qn8'		:	qn8low[x] if qn8low is not None else None,
								 'id'		:	qn_list_low[x],
								 'qnstrfmt'	:	qnstrfmt,			
								}
	
	#do it again to fill in energy levels that were upper states and didn't get hit
	for x in range(len(frequency)):
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up] is None:
			#calculate the energy.  Move the transition from MHz -> cm-1 -> K
			freq_cm = (frequency[x]*1E6/ccm)
			freq_K = freq_cm / kcm
			level_dict[qnstr_up] = {'energy'	:	elow[x] + freq_K,
									 'g'		:	gup[x],
									 'g_flag'	:	False,
									 'qn1'		:	qn1up[x] if qn1up is not None else None,
									 'qn2'		:	qn2up[x] if qn2up is not None else None,
									 'qn3'		:	qn3up[x] if qn3up is not None else None,
									 'qn4'		:	qn4up[x] if qn4up is not None else None,
									 'qn5'		:	qn5up[x] if qn5up is not None else None,
									 'qn6'		:	qn6up[x] if qn6up is not None else None,
									 'qn7'		:	qn7up[x] if qn7up is not None else None,
									 'qn8'		:	qn8up[x] if qn8up is not None else None,
									 'id'		:	qn_list_up[x],
									 'qnstrfmt'	:	qnstrfmt,			
									}			
	
	#go grab the degeneracies	
	for x in range(len(frequency)):	
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up]['g'] is None:
			level_dict[qnstr_up]['g'] = gup[x]
			
	#now go through and fill any degeneracies that didn't get hit (probably ground states)
	#assume it's just 2J+1.  Set the flag for a calculated degeneracy to True.
	for x in level_dict:
		if level_dict[x]['g'] is None:
			level_dict[x]['g'] = 2*level_dict[x]['qn1'] + 1	
			level_dict[x]['g_flag'] = True	

	return level_dict
	
def _make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8):
	qn_list = [qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8]
	tmp_list = [str(x).zfill(2) for x in qn_list if x != None]
	return ''.join(tmp_list)	

@njit
def _apply_vlsr(frequency,vlsr):
	'''
	Applies a vlsr shift to a frequency array.  Frequency in [MHz], vlsr in [km/s]
	'''
	return frequency - vlsr*frequency/ckm
	
def _apply_beam(freq_arr,int_arr,source_size,dish_size,return_beam=False):
	beam_size = 206265 * 1.22 * (cm/(freq_arr * 1E6)) / dish_size #get beam size in arcsec
	beam_dilution = source_size**2 / (beam_size**2 + source_size**2)
	if return_beam is False:
		return int_arr*beam_dilution
	else:
		return int_arr*beam_dilution,beam_dilution

def find_limits(freq_arr,spacing_tolerance=100,padding=25):
	'''
	Finds the limits of a set of data, including gaps over a width, determined by the
	spacing tolerance.  Adds padding to each side to allow user to change vlsr and get 
	the simulation within the right area.
	'''
	
	if len(freq_arr) == 0:
		print('The input array has no data.')
		return

	#first, calculate the most common data point spacing as the mode of the spacings
	#this won't be perfect if the data aren't uniformly sampled
	#get the differences
	diffs = np.diff(freq_arr)
	spacing = stats.mode(diffs)[0][0]
	
	gaps = np.where(abs(diffs) > spacing*spacing_tolerance)
	
	ll = np.concatenate((np.array([freq_arr[0]]),freq_arr[gaps[0][:]+1]))
	ul = np.concatenate((freq_arr[gaps[0][:]],np.array([freq_arr[-1]])))
	
	ll -= padding*ll/ckm
	ul += padding*ul/ckm
	
	return ll,ul
	
def _find_limit_idx(freq_arr,spacing_tolerance=100,padding=25):		
	'''
	Finds the indices of the limits of a set of data, including gaps over a width, 
	determined by the  spacing tolerance.  Adds padding to each side to allow user to 
	change vlsr and get the simulation within the right area.
	'''
	
	if len(freq_arr) == 0:
		print('The input array has no data.')
		return

	#first, calculate the most common data point spacing as the mode of the spacings
	#this won't be perfect if the data aren't uniformly sampled
	#get the differences
	diffs = np.diff(freq_arr)
	spacing = stats.mode(diffs)[0][0]
	
	gaps = np.where(abs(diffs) > spacing*spacing_tolerance)
	
	ll = np.concatenate((np.array([freq_arr[0]]),freq_arr[gaps[0][:]+1]))
	ul = np.concatenate((freq_arr[gaps[0][:]],np.array([freq_arr[-1]])))
	
	ll -= padding*ll/ckm
	ul += padding*ul/ckm
	
	ll = [find_nearest(freq_arr,x) for x in ll]
	ul = [find_nearest(freq_arr,x) for x in ul]
	
	return ll,ul
	
def _find_nans(arr):
	'''
	Find the start,[stop] indices where value is present in arr
	'''

	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	# here .view(np.int8) changes the np.equal output from a bool array to a 1/0 array
	new_arr = arr
	new_arr[~np.isnan(new_arr)] = 0
	new_arr[np.isnan(new_arr)] = 1
	iszero = np.concatenate(([0], np.equal(arr, 1).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	
	lls = [x[0] for x in ranges]
	uls = [x[1] for x in ranges]
	
	return lls,uls	

def find_peaks(freq_arr,int_arr,res,min_sep,is_sim=False,sigma=3,kms=True):
	'''
	'''

	if kms is True:
		max_f = np.amax(freq_arr)
		min_f = np.amin(freq_arr)
		cfreq = (max_f + min_f)/2
		v_res = res*ckm/max_f #finest velocity spacing
		v_span = (max_f - min_f) * ckm/(cfreq) #total velocity range spanned, setting cfreq at v=0.
		v_samp = np.arange(-v_span/2,v_span/2+v_res,v_res) #create a uniformly spaced velocity array
		freq_new = v_samp*cfreq/ckm + cfreq #convert it back to frequency
		int_new = np.interp(freq_new,freq_arr,int_arr,left=0.,right=0.)
		chan_sep = min_sep/v_res
	else:
		freq_new = freq_arr
		int_new = int_arr
		chan_sep = min_sep/res
	
	indices = signal.find_peaks(int_new,distance=chan_sep)

	if kms is True:
		indices = [find_nearest(freq_arr,freq_new[x]) for x in indices[0]] #if we had to re-sample things
		
	if is_sim is True:
		return np.asarray(indices)
		
	rms = get_rms(int_arr)
	indices = [x for x in indices if int_arr[x]>sigma*rms ]
	
	return np.asarray(indices)
	
def _get_res(freq_arr):
	'''
	Return the resolution of an array (best guess).
	'''
	diffs = np.diff(freq_arr)
	return stats.mode(diffs)[0][0]
	