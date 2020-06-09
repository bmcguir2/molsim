import numpy as np
from numba import njit
from molsim.constants import ccm, cm, ckm, h, k, kcm
import math
import warnings

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
	return frequency - vlsr*frequency/ckm

		