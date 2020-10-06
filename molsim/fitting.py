import numpy as np
import lmfit
from molsim.classes import Source, Simulation, Spectrum

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
		
		'size', 'dV', 'velocity', 'Tex', 'column'
		
		and for each entry, the following:
		
		'value', 'min', 'max', 'vary'
		
		'value'	: the initial guess for the variable (float)
		'min'	: the minimum value for the variable (float)
		'max'	: the maximum value for the variable (float)
		'vary'	: True or False for whether it is allowed to vary (True) or is fixed (False)
		
	params: dict
		Non-variable parameters needed for simulation.  Default values are given.
				
		'res'			:	0.0014, #float (MHz)
		'll'			:	[float('-inf')], #list (MHz)
		'ul'			:	[float('inf')], #list (MHz)
		'line_profile'	:	'Gaussian',	#str; must be known line profile method.
		'units'			:	'K'; #str can be 'K', 'mK', 'Jy/beam'
		
	method: str
		The method to use; must be from those allowed by lmfit.  Defaults to 'leastsq' 
		which is the Levenberg-Marquardt minimization.
	
	'''
	
	#set up the internal parameters dictionary
	int_params = { 	'res'			:	0.0014,
					'll'			:	[float('-inf')],
					'ul'			:	[float('inf')],
					'line_profile'	:	'Gaussian',
					'units'			:	'K',
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
	
	def residual(params, x, obs, mol0, ll0, ul0, line_profile0, res0, units):
	
		parvals = params.valuesdict()
		size = parvals['size']
		dV = parvals['dV']
		velocity = parvals['velocity']
		Tex = parvals['Tex']
		column = parvals['column']
	
		#generate a source object
		source = Source(size = size,
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
							res = res0,
							use_obs = True,
							units = units)
		
		return np.array(obs.spectrum.Tb - sim.spectrum.int_profile)
	
	results = lmfit.minimize(residual, params, method=method, args=(obs.spectrum.frequency, obs, mol, int_params['ll'], int_params['ul'], int_params['line_profile'], int_params['res'], int_params['units']))

	return results