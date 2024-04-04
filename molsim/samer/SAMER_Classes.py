__all__ = ['Fitting_Variables', 'Pixel', 'Molecule_Parameters']

class Fitting_Variables(object):

	#Class to hold all of the variables that one would want to set before beginning the fit
	def __init__(self, source=None, spectra_dir=None, output_dir=None, x_size=None, y_size=None, observatory=None, 
				 shared_tex=None, shared_dV=None, shared_vlsr=None, tex_calculation=None, dV_calculation=None, molecule_dict=None, 
				 initial_tex=None, initial_dV=None, initial_vlsr=None, initial_nt=None, 
				 tex_flex=None, dV_flex=None, vlsr_flex=None, nt_flex=None, 
				 cont_key=None, nt_flex_pixel_1=None, custom_params_dict=None, 
				 exclusion=None, exclusion_dict=None, exclusion_nt=None, 
				 min_sep=None, sigma=None, n_chans=None, n_bins=None):

		self.source = source #The name of the source to be fit
		self.spectra_dir = spectra_dir #The directory where the outputs from extract_spectra() will be stored
		self.output_dir = output_dir #The directory where the outputs from fit_spectra() will be stored
		self.x_size = x_size #The number of pixels along the x-axis
		self.y_size = y_size #The number of pixels along the y-axis
		self.observatory = observatory #The Observatory object to use for the simulations
		self.shared_tex = shared_tex #Whether the same tex is used for all molecules (assumption of LTE)
		self.shared_dV = shared_dV #Whether the same dV is used for all molecules
		self.shared_vlsr = shared_vlsr #Whether the same vlsr is used for all molecules
		self.tex_calculation = tex_calculation #Which technique to be used to calculate tex (optically_thick, minimizer)
		#Might need another variable to set the % threshold to use for optically-thick tex calculations
		self.dV_calculation = dV_calculation #Which technique to be used to calculate dV (histogram, minimizer)
		self.molecule_dict = molecule_dict #Dictionary where molecule names are keys corresponding to list of Molecule objects
		self.initial_tex = initial_tex #The initial guess for tex when running the first pixel
		self.initial_dV = initial_dV #The initial guess for dV when running the first pixel
		self.initial_vlsr = initial_vlsr #The initial guess for the vlsr when running the first pixel
		self.initial_nt = initial_nt #The initial guess for the (log) column density when running the first pixel
		self.tex_flex = tex_flex #The default amount tex is allowed to vary in either direction from the initial value
		self.dV_flex = dV_flex #The default amount dV is allowed to vary in either direction from the initial value
		self.vlsr_flex = vlsr_flex #The default amount vlsr is allowed to vary in either direction from the initial value
		self.nt_flex = nt_flex #The default amount the column density is allowed to vary in either direction from the initial value (in log space)
		self.cont_key = cont_key #Iterable that tells gen_continuum() how many times to reuse each continuum value in case it doesn't match the number of spectral windows (as defined by find_limits())
		self.nt_flex_pixel_1 = nt_flex_pixel_1 #The amount that the column density is allowed to vary on the first pixel in log space (could be relevant when initializing all molecules with the same nt)
		self.custom_params_dict = custom_params_dict #Dictionary containing any customized initial parameters for the molecules
		self.exclusion = exclusion #Whether to utilize any kind of exclusion on the observations
		self.exclusion_dict = exclusion_dict #Dictionary containing the multiplicative factor to apply to the mean background temperature for excluding specific transitions of the molecules in the model or 'exclude' for molecules to exclude in their entirety
		self.exclusion_nt = exclusion_nt #The (log) column density to use for the simulations used to exclude observations
		#The following variables should only be set if setting dV_calculation to 'histogram'
		self.min_sep = min_sep #The minimum separation between peaks (in MHz) during the peak finding process
		self.sigma = sigma #The sigma cutoff for detecting peaks by the peakfinder (i.e. 3 for 3 sigma, etc.)
		self.n_chans = n_chans #The number of channels to include on either side of the central channel when fitting Gaussians to the peaks
		self.n_bins = n_bins #The number of bins to include in the histogram of FWHMs

		return

class Pixel(object):

	def __init__(self, pixel=None, coords=None, observation=None, obs_x=None, obs_y=None, continuum=None, ra=None, dec=None,
				 molecule_params=None, exclusion_dict=None, excluded_channels=None, rms_noise=None,
				 prev_pixel=None):

		self.pixel = pixel #The pixel coordinates as a string, e.g. 'X_Y'
		self.coords = coords #The pixel coordinates as a list, e.g. [0,1]
		self.observation = observation #The Observation object associated with this Pixel
		if self.observation is not None:
			self.obs_x = self.observation.spectrum.frequency #Array of frequency of observations at this Pixel
			self.obs_y = self.observations.spectrum.Tb #Array of intensity of observations at this Pixel
		self.continuum = continuum #Continuum object associated with this Pixel
		self.ra = ra #RA coordinate for this Pixel
		self.dec = dec #Dec coordinate for this Pixel
		self.molecule_params = molecule_params #The fit values for each parameter. Dictionary with keys corresponding to each molecule in model; associated value is the Molecule_Parameters object
		self.exclusion_dict = exclusion_dict #Dictionary with keys corresponding to each molecule with any emission excluded from observations
		self.excluded_channels = excluded_channels #Mask to pass to .obs_x and .obs_y according to .excluded_molecules
		self.rms_noise = rms_noise #rms noise value for the observations
		self.prev_pixel = prev_pixel #The pixel used as the starting position for this Pixel

		return
		
class Molecule_Parameters(object):

	def __init__(self, tex=None, dV=None, vlsr=None, nt=None,
				 init_tex=None, init_dV=None, init_vlsr=None, init_nt=None,
				 tex_flex=None, dV_flex=None, vlsr_flex=None, nt_flex=None, min_nt=None, max_nt=None,
				 associated_molecules=None, eup_threshold=None, tau_threshold=None):

		self.tex = tex #Excitation temperature (stored separately for each molecule, but will be equivalent for all molecules in model)
		self.dV = dV #Linewidth, may or may not be equivalent for all molecules
		self.vlsr = vlsr #Velocity, may or may not be equivalent for all molecules
		self.nt = nt #Log column density, unique for each molecule
		self.init_tex = init_tex #The initial guess for the excitation temperature
		self.init_dV = init_dV #The initial guess for the linewidth
		self.init_vlsr = init_vlsr #The initial guess for the velocity
		self.init_nt = init_nt #The initial guess for the column density
		self.tex_flex = tex_flex #Flexibility on the excitation temperature when moving to next pixel - default value stored in Fitting_Variables
		self.dV_flex = dV_flex #Flexibility on the linewidth when moving to next pixel - default value stored in Fitting_Variables
		self.vlsr_flex = vlsr_flex #Flexibility on the velocity when moving to next pixel - default value stored in Fitting_Variables
		self.nt_flex = nt_flex #Flexibility on the column density when moving to next pixel - default value stored in Fitting_Variables
		self.min_nt = min_nt #The lower limit on the allowed column density during the fit
		self.max_nt = max_nt #The upper limit on the allowed column density during the fit
		self.associated_molecules = associated_molecules #List of Molecule objects that share all parameters of this Molecule_Parameters instance - likely used for vib. states
		self.eup_threshold = eup_threshold #Upper state energy threshold to apply when excluding channels from the observations
		self.tau_threshold = tau_threshold #Opacity threshold to apply when excluding channels from the observations

		return