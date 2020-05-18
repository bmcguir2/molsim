import numpy as np
import math
import warnings

def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx 

def _trim_arr(arr,limits,key_arr=None,return_idxs=False,idxs=None):
	'''
	Trims the input array to the limits specified.  Optionally, will get indices from 
	the key_arr for trimming instead.
	'''
	
	if idxs is not None:
		return np.concatenate([arr[idxs[x1]:idxs[x2]] for x1,x2 in zip(range(0,len(idxs),2),range(1,len(idxs),2))])
	
	if key_arr is None:
		idxs = [find_nearest(arr,x) for x in limits]
	else:
		idxs = [find_nearest(key_arr,x) for x in limits]

	if return_idxs is False:
		return np.concatenate([arr[idxs[x1]:idxs[x2]] for x1,x2 in zip(range(0,len(idxs),2),range(1,len(idxs),2))])
	else:
		return np.concatenate([arr[idxs[x1]:idxs[x2]] for x1,x2 in zip(range(0,len(idxs),2),range(1,len(idxs),2))]),idxs