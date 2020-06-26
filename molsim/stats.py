import numpy as np
import warnings
from numba import njit

@njit
def get_rms(y,sigma=3):
	tmp_y = np.copy(y)
	i = np.nanmax(tmp_y)
	rms = np.sqrt(np.nanmean(np.square(tmp_y)))
	
	while i > sigma*rms:
		tmp_y = tmp_y[tmp_y<sigma*rms]
		rms = np.sqrt(np.nanmean(np.square(tmp_y)))
		i = np.nanmax(tmp_y)

	return rms