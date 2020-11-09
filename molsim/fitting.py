import numpy as np
import lmfit
from molsim.classes import Source, Simulation, Spectrum, Continuum
from molsim.utils import find_peaks, _get_res
from molsim.constants import ckm

def do_lsf(obs, mol, fit_vars, params=None, method='leastsq'):

	'''
	Takes an observation and a molecules and does a least squares fit, using the parameters
	specified.
	
	Attributes
	----------
	
	obs: Observation object
		Must contain the spectrum to be fit as a Spectrum object.  Will use the 'frequency'
		and 'Tb' variables in Spectrum.
		
	mol: Molecule object
		
	fit_vars: dict of dicts
		The variables that will be fit.  Must contain all of the following:
		
		'dV', 'velocity', 'Tex', 'column'
		
		may contain these, with the associated defaults:
		
		'size' (defaults to 1E20, does not vary)
		'Tbg' (defaults to using 2.7 K or a provided Continuum)
		
		and for each entry, the following:
		
		'value', 'min', 'max', 'vary'
		
		'value'	: the initial guess for the variable (float)
		'min'	: the minimum value for the variable (float)
		'max'	: the maximum value for the variable (float)
		'vary'	: True or False for whether it is allowed to vary (True) or is fixed (False)
		
	params: dict
		Non-variable parameters needed for simulation.  Default values are given.
				
		'll'			:	[float('-inf')], #list (MHz)
		'ul'			:	[float('inf')], #list (MHz)
		'line_profile'	:	'Gaussian',	#str; must be known line profile method.
		'units'			:	'K'; #str can be 'K', 'mK', 'Jy/beam'
		'continuum'		:	None, #a continuum object; 2.7K thermal will be used if not provided
		
	method: str
		The method to use; must be from those allowed by lmfit.  Defaults to 'leastsq' 
		which is the Levenberg-Marquardt minimization.
	
	'''
	
	#set up the internal parameters dictionary
	int_params = {  'll'			:	[float('-inf')],
					'ul'			:	[float('inf')],
					'line_profile'	:	'Gaussian',
					'units'			:	'K',
					'continuum'		:	None,
					}					
					
	#override defaults with user-supplied values; warn user if they've mistyped something
	for x in params:
		if x not in int_params.keys():
			print(f'WARNING: Parameter "{x}" in input parameters dictionary not a recongized value and has not been used.')
		else:
			int_params[x] = params[x]			
	
	params = lmfit.Parameters()
	for x in fit_vars:
		params.add(	x,
					value = fit_vars[x]['value'],
					min = fit_vars[x]['min'],
					max = fit_vars[x]['max'],
					vary = fit_vars[x]['vary'],
					)
	parvals = params.valuesdict()
	if 'Tbg' in parvals:
		pass
	else:
		params.add(	'Tbg',
					value = 2.7,
					min = 2.7,
					max = 2.71,
					vary = False)
	if 'size' in parvals:
		pass
	else:
		params.add(	'size',
					value = 1E20,
					min = 1E19,
					max = 1E21,
					vary = False)
	
	def residual(params, x, obs, mol0, ll0, ul0, line_profile0, units, continuum):
	
		parvals = params.valuesdict()
		size = parvals['size']
		dV = parvals['dV']
		velocity = parvals['velocity']
		Tex = parvals['Tex']
		column = parvals['column']
		Tbg = parvals['Tbg']
		
		if continuum is not None:
			pass
		else:
			continuum = Continuum(params=Tbg)
	
		#generate a source object
		source = Source(continuum = continuum,
						size = size,
						dV = dV,
						velocity = velocity,
						Tex = Tex,
						column = column,
						)
					
		#create a simulation
		sim = Simulation(	mol = mol0,
							ll = ll0,
							ul = ul0,
							observation = obs,
							source = source,
							line_profile = line_profile0,
							use_obs = True,
							units = units)
		
		#return_sims.append(sim)
		return np.array(np.abs(obs.spectrum.Tb - sim.spectrum.int_profile))
	
	results = lmfit.minimize(residual, params, method=method, args=(obs.spectrum.frequency, obs, mol, int_params['ll'], int_params['ul'], int_params['line_profile'], int_params['units'], int_params['continuum']))

	return results
	
def find_fit_limits(freq_arr,int_arr,dV,min_sep,spread,sigma=3,kms=True):
	
	'''
	Takes an input spectrum, finds the peaks, and then spits out limits that provide spread*dV on either side of the peaks for the fitting to be done.
	'''
	
	#find the peak indices
	peak_idx = find_peaks(	freq_arr = freq_arr,
							int_arr = int_arr,
							res = _get_res(freq_arr),
							min_sep = min_sep,
							is_sim = True,
							sigma = sigma,
							kms = kms,
							)
							
	#convert peak_idx into frequencies
	freqs = freq_arr[peak_idx]
	
	#convert those into ll and ul
	ll = []
	ul = []
	for x in freqs:
		span = spread*dV*x/ckm
		ll.append(x-span)
		ul.append(x+span)		
	
	return ll,ul
	
	
	
	
		