import numpy as np
from numba import njit
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

		