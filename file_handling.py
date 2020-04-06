import numpy as np
from molsim.constants import ccm, cm, ckm, h, k, kcm

def _read_freq_int(filein):
	'''Reads in a two column frequency intensity file and returns the numpy arrays	'''
	
	frequency = []
	intensity = []
	
	with open(filein, 'r') as input:
		for line in input:
			frequency.append(np.float(line.split()[0].strip()))
			intensity.append(np.float(line.split()[1].strip()))
	
	frequency = np.asarray(frequency)
	intensity = np.asarray(intensity)
			
	return frequency,intensity		