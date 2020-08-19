import numpy as np
from numba import njit
import math
from molsim.constants import ccm, cm, ckm, h, k, kcm 
from molsim.stats import get_rms
from molsim.utils import _trim_arr, find_nearest, _make_gauss, _apply_vlsr, _apply_beam
from molsim.file_io import _read_txt, _read_xy
from scipy.interpolate import interp1d
from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

class Workspace(object):

	'''
	The workspace class stores all of the information for an active processing session.
	
	Example
	-------
	
		> ws = workspace()
		
	Notes
	-----
	
	
	
	Attributes
	----------
	
		
		
	'''
	
	def __init__(self):
	
		return
			
class Catalog(object):

	'''
	The catalog class holds all the spectroscopic data for a single molecule.
	''' 		
	
	def __init__(
					self,
					catdict = None, #if attributes are to be read in from a pre-made 
									#dictionary
					catid = None, #a unique catalog id
					molecule = None, #molecule name
					frequency = None, #frequencies [MHz]
					freq_err = None, #errors on frequencies [MHz]
					measured = None, #set to True if the line is known in the laboratory, 
									 #False otherwise
					logint = None, #logarithmic intensities [log10(nm^2 MHz)]
					sijmu = None, #linestrength times the dipole moment squared [Debye^2]
					sij = None, #intrinsic linestrength (unitless)
					aij = None, #einstein A-coefficients [s^-1]
					man_int = None, #manually entered intensities, not to be used by 
									#calculations
					types = None, #transition types
					dof = None, #degree of freedom (unitless)
					elow = None, #lower state energy [K]
					eup = None, #upper state energy [K]
					glow = None, #lower state degeneracy (unitless)
					gup = None, #upper state degeneracy (unitless)
					tag = None, #spcat tag
					qnformat = None, #spcat quantum number format string
					qn1up = None, #upper state principle quantum number 1
					qn2up = None, #upper state quantum number 2
					qn3up = None, #upper state quantum number 3
					qn4up = None, #upper state quantum number 4
					qn5up = None, #upper state quantum number 5
					qn6up = None, #upper state quantum number 6
					qn7up = None, #upper state quantum number 7
					qn8up = None, #upper state quantum number 8
					qnup_str = None, #upper state quantum number string
					qn1low = None, #lower state principle quantum number 1
					qn2low = None, #lower state quantum number 2
					qn3low = None, #lower state quantum number 3
					qn4low = None, #lower state quantum number 4
					qn5low = None, #lower state quantum number 5
					qn6low = None, #lower state quantum number 6
					qn7low = None, #lower state quantum number 7
					qn8low = None, #lower state quantum number 8	
					qnlow_str = None, #lower state quantum number string
					qnstr_fmt = None, #commands for formatting this molecule's qns
					version = None, #version of this catalog in the database			
					source = None, #where the catalog came from
					last_update = None,	#when it was last updated
					contributor_name = None, #who contributed the catalog to this database
					contributor_email = None, #email of contributor
					notes = None, #other notes, including a change log
					refs = None, #literature references
				):
		
		self.catdict = catdict
		self.catid = catid 
		self.molecule = molecule
		self.frequency = frequency
		self.freq_err = freq_err
		self.measured = measured
		self.logint = logint
		self.sijmu = sijmu
		self.sij = sij
		self.aij = aij
		self.man_int = man_int
		self.types = types
		self.dof = dof
		self.elow = elow
		self.eup = eup
		self.glow = glow
		self.gup = gup
		self.tag = tag
		self.qnformat = qnformat
		self.qn1up = qn1up
		self.qn2up = qn2up
		self.qn3up = qn3up
		self.qn4up = qn4up
		self.qn5up = qn5up
		self.qn6up = qn6up
		self.qn7up = qn7up
		self.qn8up = qn8up
		self.qnup_str = qnup_str
		self.qn1low = qn1low
		self.qn2low = qn2low
		self.qn3low = qn3low
		self.qn4low = qn4low
		self.qn5low = qn5low
		self.qn6low = qn6low
		self.qn7low = qn7low
		self.qn8low = qn8low	
		self.qnlow_str = qnlow_str	
		self.qnstr_fmt = qnstr_fmt
		self.version = version
		self.source = source
		self.last_update = last_update
		self.contributor_name = contributor_name
		self.contributor_email = contributor_email
		self.notes = notes
		self.refs = refs
		
		self._unpack_catdict()
	
		return
		
	def _unpack_catdict(self):
	
		'''
		If a dictionary of data was provided, go ahead and use that to unpack things.
		'''
	
		if self.catdict is not None:
			if all(['catid' in self.catdict, self.catid is None]):	
				self.catid = self.catdict['catid'] 
			if all(['molecule' in self.catdict, self.molecule is None]):
				self.molecule = self.catdict['molecule']
			if all(['frequency' in self.catdict, self.frequency is None]):
				self.frequency = self.catdict['frequency']
			if all(['freq_err' in self.catdict, self.freq_err is None]):
				self.freq_err = self.catdict['freq_err']
			if all(['measured' in self.catdict, self.measured is None]):
				self.measured = self.catdict['measured']
			if all(['logint' in self.catdict, self.logint is None]):
				self.logint = self.catdict['logint']
			if all(['sijmu' in self.catdict, self.sijmu is None]):
				self.sijmu = self.catdict['sijmu']
			if all(['sij' in self.catdict, self.sij is None]):
				self.sij = self.catdict['sij']
			if all(['aij' in self.catdict, self.aij is None]):
				self.aij = self.catdict['aij']
			if all(['man_int' in self.catdict, self.man_int is None]):
				self.man_int = self.catdict['man_int']
			if all(['types' in self.catdict, self.types is None]):
				self.types = self.catdict['types']
			if all(['dof' in self.catdict, self.dof is None]):
				self.dof = self.catdict['dof']
			if all(['elow' in self.catdict, self.elow is None]):
				self.elow = self.catdict['elow']
			if all(['eup' in self.catdict, self.eup is None]):
				self.eup = self.catdict['eup']
			if all(['glow' in self.catdict, self.glow is None]):
				self.glow = self.catdict['glow']
			if all(['gup' in self.catdict, self.gup is None]):
				self.gup = self.catdict['gup']
			if all(['tag' in self.catdict, self.tag is None]):
				self.tag = self.catdict['tag']
			if all(['qnformat' in self.catdict, self.qnformat is None]):
				self.qnformat = self.catdict['qnformat']
			if all(['qn1up' in self.catdict, self.qn1up is None]):
				self.qn1up = self.catdict['qn1up']
			if all(['qn2up' in self.catdict, self.qn2up is None]):
				self.qn2up = self.catdict['qn2up']
			if all(['qn3up' in self.catdict, self.qn3up is None]):
				self.qn3up = self.catdict['qn3up']
			if all(['qn4up' in self.catdict, self.qn4up is None]):
				self.qn4up = self.catdict['qn4up']
			if all(['qn5up' in self.catdict, self.qn5up is None]):
				self.qn5up = self.catdict['qn5up']
			if all(['qn6up' in self.catdict, self.qn6up is None]):
				self.qn6up = self.catdict['qn6up']
			if all(['qn7up' in self.catdict, self.qn7up is None]):
				self.qn7up = self.catdict['qn7up']
			if all(['qn8up' in self.catdict, self.qn8up is None]):
				self.qn8up = self.catdict['qn8up']
			if all(['qnup_str' in self.catdict, self.qnup_str is None]):
				self.qnup_str = self.catdict['qnup_str']			
			if all(['qn1low' in self.catdict, self.qn1low is None]):
				self.qn1low = self.catdict['qn1low']
			if all(['qn2low' in self.catdict, self.qn2low is None]):
				self.qn2low = self.catdict['qn2low']
			if all(['qn3low' in self.catdict, self.qn3low is None]):
				self.qn3low = self.catdict['qn3low']
			if all(['qn4low' in self.catdict, self.qn4low is None]):
				self.qn4low = self.catdict['qn4low']
			if all(['qn5low' in self.catdict, self.qn5low is None]):
				self.qn5low = self.catdict['qn5low']
			if all(['qn6low' in self.catdict, self.qn6low is None]):
				self.qn6low = self.catdict['qn6low']
			if all(['qn7low' in self.catdict, self.qn7low is None]):
				self.qn7low = self.catdict['qn7low']
			if all(['qn8low' in self.catdict, self.qn8low is None]):
				self.qn8low = self.catdict['qn8low']
			if all(['qnlow_str' in self.catdict, self.qnlow_str is None]):
				self.qnlow_str = self.catdict['qnlow_str']				
			if all(['qnstr_fmt' in self.catdict, self.qnstr_fmt is None]):
				self.qnstr_fmt = self.catdict['qnstr_fmt']				
			if all(['version' in self.catdict, self.version is None]):
				self.version = self.catdict['version']
			if all(['source' in self.catdict, self.source is None]):
				self.source = self.catdict['source']
			if all(['last_update' in self.catdict, self.last_update is None]):
				self.last_update = self.catdict['last_update']
			if all(['contributor_name' in self.catdict, self.contributor_name is None]):
				self.contributor_name = self.catdict['contributor_name']
			if all(['contributor_email' in self.catdict, self.contributor_email is None]):
				self.contributor_email = self.catdict['contributor_email']
			if all(['notes' in self.catdict, self.notes is None]):
				self.notes = self.catdict['notes']
			if all(['refs' in self.catdict, self.refs is None]):
				self.refs = self.catdict['refs']

		
		return
		
	def _set_sijmu_aij(self,Q):
		eq1 = 2.40251E4 * 10**(self.logint) * Q.qrot(300) * self.frequency **-1
		eq2 = np.exp(-self.elow/300) - np.exp(-self.eup/300)
		self.sijmu = eq1/eq2
		self.aij = 1.16395E-20*self.frequency**3*self.sijmu/self.gup

		return	
		
	def export_cat(self,fileout,catformat='molsim'):
		'''
		Exports a catalog to an output file.  If catformat is set to 'molsim' it outputs a
		file that will read into molsim much faster in the future.  If it is set to 
		'spcat' it will output in spcat format.
		'''
		
		if catformat == 'molsim':
			np.savez_compressed(fileout,
								catdict = self.catdict,
								catid = self.catid ,
								molecule = self.molecule,
								frequency = self.frequency,
								freq_err = self.freq_err,
								measured = self.measured,
								logint = self.logint,
								sijmu = self.sijmu,
								sij = self.sij,
								aij = self.aij,
								man_int = self.man_int,
								types = self.types,
								dof = self.dof,
								elow = self.elow,
								eup = self.eup,
								glow = self.glow,
								gup = self.gup,
								tag = self.tag,
								qnformat = self.qnformat,
								qn1up = self.qn1up,
								qn2up = self.qn2up,
								qn3up = self.qn3up,
								qn4up = self.qn4up,
								qn5up = self.qn5up,
								qn6up = self.qn6up,
								qn7up = self.qn7up,
								qn8up = self.qn8up,
								qnup_str = self.qnup_str,
								qn1low = self.qn1low,
								qn2low = self.qn2low,
								qn3low = self.qn3low,
								qn4low = self.qn4low,
								qn5low = self.qn5low,
								qn6low = self.qn6low,
								qn7low = self.qn7low,
								qn8low = self.qn8low	,
								qnlow_str = self.qnlow_str,
								qnstr_fmt = self.qnstr_fmt,
								version = self.version,
								source = self.source,
								last_update = self.last_update,
								contributor_name = self.contributor_name,
								contributor_email = self.contributor_email,
								notes = self.notes,
								refs = self.refs
							)
		
		return	
	
class Level(object):

	'''
	The level class holds all the information for an energy level.
	''' 		
	
	def __init__(
					self,
					energy = None,	#energy of the level [K]
					g = None, #degeneracy of this level
					g_flag = False, #is the degeneracy calculated by the program
					qn1 = None,	#qn1
					qn2 = None, #qn2
					qn3 = None,	#qn3
					qn4 = None, #qn4
					qn5 = None, #qn5
					qn6 = None, #qn6
					qn7 = None, #qn7
					qn8 = None, #qn8
					nqns = None, #number of quantum numbers
					id = None, #unique ID for this level
					trans = None, #IDs of all transitions that link to this level
					ltrans = None, #IDs of all transitions for which this is the lower state
					utrans = None, #IDs of all transitions for which this is the upper state
					qnstrfmt = None, #quantum number string format
					qnstr = None, #complete quantum number string
					qnstr_tex = None, #complete quantum number string with LaTeX code
					molid = None, #ID of the molecule this energy level belongs to
					mol = None, #the actual associated molecule class
				):
				
		self.energy = energy
		self.g = g
		self.g_flag = g_flag
		self.qn1 = qn1
		self.qn2 = qn2
		self.qn3 = qn3
		self.qn4 = qn4
		self.qn5 = qn5
		self.qn6 = qn6
		self.qn7 = qn7
		self.qn8 = qn8
		self.nqns = nqns
		self.id = id
		self.trans = trans
		self.ltrans = ltrans
		self.utrans = utrans
		self.qnstrfmt = qnstrfmt
		self.qnstr = qnstr
		self.qnstr_tex = qnstr_tex
		self.molid = molid
		self.mol = mol		
		
		return	
		
class Transition(object):

	'''
	The transition class holds all the information for a single transition
	''' 		
	
	def __init__(
					self,
					frequency = None,	#frequency of transition in [MHz]
					freq_err = None, #uncertainty in transition [MHz]
					measured = None, #set to True if line is known in the Laboratory
					elow = None,	#energy of the lower level in [K]
					eup = None,	#energy of the upper level in [K]
					glow = None, #degeneracy of the lower level
					gup = None, #degeneracy of the upper level
					qn1up = None, #upper state principle quantum number 1
					qn2up = None, #upper state quantum number 2
					qn3up = None, #upper state quantum number 3
					qn4up = None, #upper state quantum number 4
					qn5up = None, #upper state quantum number 5
					qn6up = None, #upper state quantum number 6
					qn7up = None, #upper state quantum number 7
					qn8up = None, #upper state quantum number 8
					qnup_str = None, #string of upper states smashed together
					qn1low = None, #lower state principle quantum number 1
					qn2low = None, #lower state quantum number 2
					qn3low = None, #lower state quantum number 3
					qn4low = None, #lower state quantum number 4
					qn5low = None, #lower state quantum number 5
					qn6low = None, #lower state quantum number 6
					qn7low = None, #lower state quantum number 7
					qn8low = None, #lower state quantum number 8
					qnlow_str = None, #string of lower states mashed together
					qnstr_formatted = None, #formatted quantum number string	
					nqns = None, #number of quantum numbers
					id = None, #unique ID for this transition
					sijmu = None, #sijmu2 [debye^2]
					sij = None, #sij [unitless]
					aij = None, #aij [s^-1]
					logint = None, #logarithmic intensity [log10(nm^2 MHz)]
					type = None, #transition type
					upper_level = None, #Level object of upper level
					lower_level = None, #Level object of lower level
					mol = None, #the molecule object
					catalog = None, #the catalog object
				):
				
		self.frequency = frequency
		self.freq_err = freq_err
		self.measured = measured
		self.elow = elow
		self.eup = eup
		self.glow = glow
		self.gup = gup
		self.qn1up = qn1up
		self.qn2up = qn2up
		self.qn3up = qn3up
		self.qn4up = qn4up
		self.qn5up = qn5up
		self.qn6up = qn6up
		self.qn7up = qn7up
		self.qn8up = qn8up
		self.qnup_str = qnup_str
		self.qn1low = qn1low
		self.qn2low = qn2low
		self.qn3low = qn3low
		self.qn4low = qn4low
		self.qn5low = qn5low
		self.qn6low = qn6low
		self.qn7low = qn7low
		self.qn8low = qn8low
		self.qnlow_str = qnlow_str
		self.qnstr_formatted = qnstr_formatted
		self.nqns = nqns
		self.id = id
		self.sijmu = sijmu
		self.sij = sij
		self.aij = aij
		self.logint = logint
		self.type = type
		self.upper_level = upper_level
		self.lower_level = lower_level
		self.mol = mol		
		self.catalog = catalog
		
		return		
		
class Molecule(object):		
		
	'''
	The molecule class holds all the information for a single molecule.  Note that this
	does not include anything about an observed population of this molecule.  It is 
	only the physical properties of a molecule.  Things like column density and temp
	are part of the <classname> class.
	''' 		
	
	def __init__(
					self,
					id = None, #unique ID for this molecule
					name = None, #common name (str)
					formula = None, #formula (str)
					elements = None, #a dictionary containing the elemental composition
					mass = None, #molecular mass [amu]
					A = None, #A rotational constant [MHz]
					B = None, #B rotational constant [MHz]
					C = None, #C rotational constant [MHz]
					muA = None, #A component of the dipole moment [Debye]
					muB = None, #B component of the dipole moment [Debye]
					muC = None, #C component of the dipole moment [Debye]
					mu = None, #total dipole moment
					qpart = None, #partition function object
					catalog = None, #catalog object for this molecule
					trans = None, #list of transition objects for this molecule
					levels = None, #list of energy level objects for this molecule
				):
				
		self.id = id
		self.name = name
		self.formula = formula
		self.elements = elements
		self.mass = mass
		self.A = A
		self.B = B
		self.C = C
		self.muA = muA
		self.muB = muB
		self.muC = muC
		self.mu = mu
		self.qpart = qpart
		self.catalog = catalog
		self.trans = trans
		self.levels = levels

		
		return
		
	def qrot(self,T):
		return self.qpart.qrot(T)
	
	def qvib(self,T):
		return self.qpart.qvib(T)	
		
	def q(self,T):
		return self.qpart.q(T)
		 							
class PartitionFunction(object):		
		
	'''
	The partitionfunction class holds all of the info necessary to calculate a partion
	function for a molecule.  The calculation method is flexible.  
	
	Notes
	-----
	If more than one method of calculation is specified, the program will default to 
	using a specified functional form first, then a list of values for interpolation,
	then explicit state counting.  
	
	If no methods are specified, the partition function is set to 1. for all values.
	
	To specify a functional form, provide values for 'form' and 'params'.
	
	To specify a set of values for interpolation, provide arrays for 'temps' and 'vals'.
	
	To specify explicit state counting, either provide arrays of degeneracies ('gs') and
	'energies' to be used, or a catalog object for 'cat' to pull these values from. 
	Optionally, a sigma value can be specified with 'sigma'.
	
	Vibrational states can be optionally specified (see below).
	
	For record keeping, you may enter 'notes'.
	
	Attributes
	----------
	form : str
		The type of functional form you wish to use to calculate.  Currently supported
		options are:
			'poly' or 'polynomial' for a polynomial of arbitrary order, including linear
			'pow' or 'power' for a power law (Q = A*T^pow + B)
			'rotcons' to estimate from the rotational constants via Gordy & Cook
			
	params : list
		The parameters for your functional form calculation.  Should be a list of floats.
		Formatting is specified as follows:
			
			polynomial
			----------
				A list in increasing order of T.  For example:
					[1.4, -3.2, 6.5]
				produces the functional form:
					Qr = 1.4 - 3.2*T + 6.5*T^2
					
			power
			-----
				[A, B, pow] produces:
					Qr = A*T^pow + B
					
			rotcons
			-------
				All constants must be expressed in units of MHz
				[B] for a linear molecule
				[A, B, [sigma]] for a symmetric top.  sigma is optional and defaults to 1
				[A, B, C, [sigma]] for a non-linear molecule, sigma is optional and defaults to 1
				
				WARNING: This produces at best an OK approximation.  Often use of this 
				formulation results in partition functions that are incorrect by factors 
				of a few to an order of magnitude for all but the simplest of cases.
				
				Examples:
				
					[42000] invokes:
						Qr = kT/hB (Eq. 3.64 in Gordy & Cook)
							and produces:
						Qr = kT/h(4200)
					
					[2500, 1500] invokes:
						Qr = (5.34E6/sigma)*sqrt((T^3/(B^2*A))) (Eq. 3.68 in Gordy & Cook)
							and produces:
						Qr = (5.34E6/1)*sqrt((T^3/((1500)^2*(2500))))
						
					[2500, 1500, [2]] invokes:	
						Qr = (5.34E6/sigma)*sqrt((T^3/(B^2*A))) (Eq. 3.68 in Gordy & Cook)
							and produces:
						Qr = (5.34E6/2)*sqrt((T^3/((1500)^2*(2500))))	
						
					[2500, 1500, 750] invokes:
						Qr = (5.34E6/sigma)*sqrt((T^3/(ABC))) (Eq. 3.69 in Gordy & Cook)
							and produces:
						Qr = (5.34E6/1)*sqrt((T^3/(750 * 1500 * 2500)))	
						
					[2500, 1500, 750, [2]] invokes:
						Qr = (5.34E6/sigma)*sqrt((T^3/(ABC))) (Eq. 3.69 in Gordy & Cook)
							and produces:
						Qr = (5.34E6/2)*sqrt((T^3/(750 * 1500 * 2500)))	
						
	temps : array
		A numpy array of the temperatures to use in interpolating partition function 
		values. Units are Kelvin.  Interpolation is linear between points.
		
	vals : array
		A numpy array of the partition function values at the temperatures specified
		in 'temps'.  Interpolation is linear between points.
		
	mol : Molecule object
		A Molecule object that contains degeneracies and energies used for state counting
		
	gs : array
		A numpy array of the degeracies of each state for state counting
		
	energies : array
		A numpy array of the energies of each state for state counting
		
	sigma : int
		A symmetry parameter optionally used to alter state counting.  Defaults to 1.		
		
	vib_states : array
		A numpy array of vibrational states.  Units of wavenumbers (cm-1).
		
	vib_is_K : bool
		Set to True if vibstates are given in Kelvin
		
	notes :	str
		Any notes for this partition function.																
				
	''' 		
	
	def __init__(
					self,
					qpart_file = None, #a file that holds this info externally
					form = None, #the functional form of the partition function
					params = None, #the parameters for that functional form
					temps = None, #a temperature array if we're going to interpolate [K]
					vals = None, #a values array if we're going to interpolate
					mol = None, #a molecule to pull gs and energies from
					gs = None, #an array of degeneracies to calculate from
					energies = None, #an array of energies to calculate from [K]
					sigma = 1, #a sigma value for state counting
					vib_states = None, #an array of vibrational states to calculate from
										#these are in [cm-1]
					vib_is_K = False, #set to true if your vibstates are in Kelvin
					notes = None, #a way to add notes
				):
		
		self.qpart_file = qpart_file		
		self.form = form
		self.params = params
		self.temps = temps
		self.vals = vals
		self.mol = mol
		self.gs = gs
		self.energies = energies
		self.sigma = sigma
		self.vib_states = vib_states
		self.vib_is_K = vib_is_K
		self.notes = notes
		self.flag = None

		self._setup()
		self._check_functional()
		
		return				
		
	def _setup(self):
		if self.qpart_file is None:
			self._initialize_flag()
		else:
			form_line = None
			vibs_read = False
			vibs_line = None
			qpart_raw = _read_txt(self.qpart_file)
			for i in range(len(qpart_raw)):
				if '#' in qpart_raw[i]:
					linesplit = qpart_raw[i].split(':')
					if 'form' in linesplit[0]:
						self.form = linesplit[1].strip()
						if self.form in ['poly','polynomial','power','pow','rotcons']:
							self.flag = 'functional'
						else: 
							self.flag = linesplit[1].strip()
						form_line = i+1
					if 'vibs' in linesplit[0]:
						vibs_read = True
						vibs_line = i+1
			if self.form == 'interpolation':
				t_arr = []
				q_arr = []
				for line in qpart_raw:
					if '#' not in line:
						t_arr.append(float(line.split()[0]))
						q_arr.append(float(line.split()[1].strip()))
				self.temps = np.array(t_arr)
				self.vals = np.array(q_arr)
			if self.form in ['pow','power']:
				self.params = [float(qpart_raw[form_line].split(',')[0].strip()),float(qpart_raw[form_line].split(',')[1].strip()),float(qpart_raw[form_line].split(',')[2].strip())]
			if self.form in ['poly', 'polynomial']:
				self.params = []
				for x in qpart_raw[form_line].split(','):
					self.params.append(float(x.strip()))
			if vibs_read is True:
				vibs = []
				for x in qpart_raw[vibs_line].split(','):
					vibs.append(float(x.strip()))
				self.vib_states = np.asarray(vibs)
				
		return
					
			
		
	def _initialize_flag(self):
	
		if sum([i is None for i in [self.form,self.temps,self.gs,self.mol]]) < 2:
			print('WARNING: More than one way has been specified to calculate this ' \
				  'partition function. Please specify only a single method.  For now, ' \
				  'a default has been chosen from the priority-ordered list:\n \t1) a ' \
				  'specified functional form\n \t2) a specified list of values to ' \
				  'interpolate \n \t3) explicit state counting from a list of ' \
				  'energies and degeneracies.')
			if self.form is not None:
				self.flag = 'functional'
				return
			if self.temps is not None:
				self.flag = 'interpolation'
				return
			if self.gs is not None:
				self.flag = 'counting'
				return
	
		if self.form is not None:
			self.flag = 'functional'
			return
		if self.temps is not None:
			self.flag = 'interpolation'
			return
		if self.gs is not None:
			self.flag = 'counting'
			return
		if self.mol is not None:
			self.flag = 'counting'
			return
			
		#if none of these were specified, there's a problem, and notify the user
		print('WARNING: Not enough information was provided to calculate a partition ' \
				'function.  A value of 1 will be used at all temperatures.')	
		self.flag = 'constant'
		
	def _check_functional(self):
	
		'''
		Checks to make sure the user specified the type of functional form and that
		it is on the list of ones the program can handle.
		'''
		
		if self.flag != 'functional':
			return
			
		functionals = ['poly','polynomial','power','pow','rotcons']
		
		if self.flag == 'functional' and self.form not in functionals:
			print('ERROR: The partition function has been specified to be calculated ' \
				  'analytically using a functional form.  The form specified is either '\
				  'not currently supported, or is unrecognized.  The form string that '\
				  'was entered was: "{}". Please choose from this list of options:\n'\
				  '\t-- "poly" or "polynomial" for any order polynmial function\n'\
				  '\t-- "pow" or "power" for a power law function\n'\
				  '\t-- "rotcons" to estimate from Gordy & Cook rotational constants '\
				  'formalism.\n Qrot is set to 1 until this is corrected.'
				)
			return False
		else:
			return True
		
	def qrot(self,T):
	
		'''
		Calculate and return the rotational partition function at temperature T
		'''
	
		#if the user specified one of the allowed functional forms
		if self.flag == 'functional':
			
			#if a correct functional form hasn't been specified, return 1
			if self._check_functional() is False:
				return 1.		
			
			#if it's a polynomial...
			if self.form in ['poly','polynomial']:
				norder = len(self.params) #get the order of the polynomial
				qrot = 0.
				for i in reversed(range(norder)):
					qrot += self.params[i]*T**i
				return qrot
					
			#if it's a power law...
			if self.form in ['pow','power']:
				return self.params[0]*T**self.params[2] + self.params[1]
				
			#if it's rotational constants
			if self.form == 'rotcons':
				#sort out if there's sigma specified or not
				if list not in [type(x) for x in self.params]: #no sigma, use defaults
					if len(self.params) == 1:
						return k*T/(h * self.params[0]*1E6)
					if len(self.params) == 2:
						return (5.34E6/1)*np.sqrt((T**3/(self.params[1]**2 * self.params[0])))
					if len(self.params) == 3:
						return (5.34E6/1)*np.sqrt((T**3/(self.params[2] * self.params[1] * self.params[0])))
			
				if list in [type(x) for x in self.params]: #use a sigma
					if len(self.params) == 3:
						return (5.34E6/self.params[2][0])*np.sqrt((T**3/(self.params[1]**2 * self.params[0])))
					if len(self.params) == 4:
						return (5.34E6/self.params[3][0])*np.sqrt((T**3/(self.params[2] * self.params[1] * self.params[0])))					
						

		#if the user provided arrays for interpolation
		if self.flag == 'interpolation':
			 f = interp1d(self.temps,self.vals,fill_value='extrapolate')
			 return f(T).tolist()
			 
		#if the user provided a catalog or gs and energies
		if self.flag == 'counting':
			#if a catalog:
			if self.mol is not None:
				gs = np.array([level.g for level in self.mol.levels])
				energies = np.array([level.energy for level in self.mol.levels])
			else:
				gs = self.gs
				energies = self.energies
			return (1/self.sigma)*np.sum(gs*np.exp(-energies/T))
	
	def qvib(self,T):
		'''
		Calculate and return the vibrational partition function at temperature T
		'''		 
		
		#if there's no states specified, return 1.
		if self.vib_states is None:
			return 1.
		
		#otherwise do the calculation and return based on the units
		if self.vib_is_K is True:
			return np.prod(np.sum(np.exp((-self.vib_states[:,np.newaxis]*np.arange(100))/(T)),axis=1))
		else:
			return np.prod(np.sum(np.exp((-self.vib_states[:,np.newaxis]*np.arange(100))/(0.695*T)),axis=1))	
		
	def q(self,T):
		'''
		Calculate and return the total partition function at a temperature T
		'''	
		
		return self.qrot(T)*self.qvib(T)
		
class Spectrum(object):

	'''
	This class stores the information for a single spectrum.
	'''		
	
	def __init__(
					self,
					freq0 = None, #unshifted frequency data
					frequency = None, #frequency data
					Tb = None, #intensity in units of [K]
					Iv = None, #intensity in units of [Jy/beam] or [Jy/sr]
					Tbg = None, #intensity of background in [K]
					Ibg = None, #intensity of background in [Jy/beam] or [Jy/sr]
					tau = None, #optical depths
					tau_profile = None, #tau with line profile applied
					freq_profile = None, #frequency of line profile data
					int_profile = None, #intensity of line profile data
					Tbg_profile = None, #background with line profile
					velocity = None, #velocity space Data
					int_sim = None, #intensity of a simulation
					freq_sim = None, #frequency of a simulation
					snr = None, #data in snr space
					id = None, #a unique ID for this spectrum
					notes = None, #notes
					name = None, #name
				):
		
		self.freq0 = freq0
		self.frequency = frequency
		self.Tb = Tb
		self.Iv = Iv
		self.Tbg = Tbg
		self.Ibg = Ibg
		self.tau = tau
		self.tau_profile = tau_profile
		self.freq_profile = freq_profile
		self.int_profile = int_profile
		self.Tbg_profile = Tbg_profile
		self.id = id
		self.notes = notes
		self.ll = np.amin(frequency) if frequency is not None else None
		self.ul = np.amax(frequency) if frequency is not None else None
		self.name = name
		self.velocity = velocity
		self.int_sim = int_sim
		self.freq_sim = freq_sim
		self.snr = snr

		return	
		

	def export_spectrum(self,fileout,format='molsim'):
		'''
		Export a spectrum and all meta data out to a file.  If format='molsim', it will be
		an npz file that is much faster for restoring later.
		'''
		
		if format == 'molsim':
			np.savez_compressed(fileout,
								freq0 = self.freq0,
								frequency = self.frequency,
								Tb = self.Tb,
								Iv = self.Iv,
								Tbg = self.Tbg,
								Ibg = self.Ibg,
								tau = self.tau,
								tau_profile = self.tau_profile,
								freq_profile = self.freq_profile,
								int_profile = self.int_profile,
								Tbg_profile = self.Tbg_profile,
								id = self.id,
								notes = self.notes,
								name = self.name,
								velocity = self.velocity,
								int_sim = self.int_sim,
								freq_sim = self.freq_sim,
								snr = self.snr,
							)
		
		return				
		
class Continuum(object):

	'''
	This class stores the information needed to provide a continuum value at any point
	'''		
	
	def __init__(
					self,
					cont_file = None, #a cont file to read parameters from
					type = 'thermal', #type of continuum to calculate
					params = [2.7], #necessary parameters
					freqs = None, #frequencies [MHz] if interpolating between points
					temps = None, #values [K] if interpolating T between points
					fluxes = None, #fluxes [Jy/beam] if interpolating Jy between points
					notes = None, #notes 
				):
				
		self.cont_file = cont_file		
		self.type = type
		self.params = params
		self.freqs = freqs
		self.temps = temps
		self.fluxes = fluxes
		self.notes = notes
		
		self._check_type()
		self._fix_params()
		
		return
		
	def _check_type(self):
		types = ['thermal','polynomial','poly','interpolation']
		if self.type not in types:
			print('WARNING: Unrecognized type ("{}") specified for continuum generation.' \
			' Will use 2.7 K CMB instead.' .format(self.type))
			self.params = [2.7]
		return
		
	def _fix_params(self):
		if isinstance(self.params,int) or isinstance(self.params,float):
			self.params = [self.params]	
		
	def Tbg(self,freq):
		'''
		Takes an input frequency numpy array [MHz] and returns a numpy array of brightness
		temperatures at those frequencies.
		'''
		
		if self.type == 'thermal':
			return np.full_like(freq,self.params[0])
				
	def Ibg(self,freq):			
		'''
		Takes an input frequency numpy array [MHz] and returns a numpy array of flux
		densities [Jy/sr] at those frequencies.
		'''
		
		if self.type == 'thermal':
			return 2*h*(freq*1E6)**3 / (cm**2 * np.exp(h*freq*1E6/(k*self.params[0])))*1E26		

class Source(object):

	'''
	This class stores the information for a single source of molecules
	'''		
	
	def __init__(
					self,
					name = None, # name
					velocity = 0., #lsr velocity [km/s]
					size = 1E20, #diameter [arcsec]
					solid_angle = None, #solid angle on the sky; pi*(size/2)^2 [arcsec^2]
					continuum = None, #a continuum object
					column = 1.E13, #column density [cm-2]
					Tex = 300., #float or numpy array of excitation temperatures [K]
					Tkin = None, #kinetic temperature of source [K]
					dV = 3., #fwhm [km/s]
					id = None, #unique id
					notes = None, #notes
				):
				
		self.name = name
		self.velocity = velocity
		self.size = size
		self.solid_angle = solid_angle
		self.continuum = continuum if continuum is not None else Continuum()
		self.column = column
		self.Tex = Tex
		self.Tkin = Tkin
		self.dV = dV
		self.id = id
		self.notes = notes
		
		return			

class Observatory(object):

	'''
	This class stores the information for a single Observatory
	'''		
	
	def __init__(
					self,
					name = None, #telescope name, str
					id = None, #unique ID
					sd = True, #is it a single dish?
					array = False, #is it an array?
					dish = 100., #dish size in meters if single dish
					synth_beam = [1.,1.], #synthesized beam bmaj,bmin in arcseconds
					loc = None, #astropy EarthLocation object 
					eta = None, #numpy array of aperture efficiency
					eta_type = 'constant', #how to calculate eta
					eta_params = [1.], #parameters for calculating eta
					atmo = None, #numpy array of atmospheric transmission in percent
				):
				
		self.name = name
		self.id = id
		self.sd = sd
		self.array = array
		self.dish = dish
		self.synth_beam = synth_beam
		self.loc = loc
		self.eta = eta
		self.eta_type = eta_type
		self.eta_params = eta_params
		self.atmo = atmo
		
		return
		
	def get_beam(self,freq):
		return 	206265 * 1.22 * ((freq*u.MHz).to(u.m, equivalencies=u.spectral()).value) / self.dish		

class Observation(object):

	'''
	This class stores the information for a observation
	'''		
	
	def __init__(
					self,
					name = None, #a name
					coords = None, #an astropy SkyCoord object		
					vlsr = None, #a nominal vlsr for the observed objection [km/s]			
					spectrum = None, #a spectrum object for this observation
					observatory = None, #an observatory object for this observation
					id = None, #a unique ID for this observation
					notes = None, #notes
				):
				
		self.name = name
		self.coords = coords		
		self.vlsr = vlsr
		self.spectrum = spectrum if spectrum is not None else Spectrum()
		self.observatory = observatory if observatory is not None else Observatory()
		self.id = id
		self.notes = notes
		
		return				

class Simulation(object):
	'''
	This class stores the information for a single simulation
	'''		
	
	def __init__(
					self,
					spectrum = None, #Spectrum object associated with this simulation
					observation = None, #Observation object associated with this simulation
					source = None, #Source object associated with this simulation
					ll = [np.float('-inf')], #lower limits
					ul = [np.float('-inf')], #lower limits
					line_profile = None, #simulate a line profile or not
					sim_width = 10, #fwhms to simulate +/- line center
					res = 10., #resolution if simulating line profiles [kHz]
					mol = None, #Molecule object associated with this simulation
					notes = None, #notes
				):
				
		self.spectrum = spectrum
		self.observation = observation
		self.source = source
		self.ll = ll
		self.ul = ul
		self.line_profile = line_profile
		self.sim_width = sim_width
		self.res = res
		self.mol = mol
		self.notes = notes
		self.beam_size = None
		self.beam_dilution = None
		self.ll_idxs = None
		self.ul_idxs = None
		self.aij = None
		self.gup = None
		self.eup = None
		
		self._set_arrays()
		self._apply_voffset()
		self._calc_tau()
		self._calc_bg()
		self._calc_Iv()
		self.spectrum.Tb = self._calc_Tb(self.spectrum.frequency,self.spectrum.tau,self.spectrum.Tbg)
		self._make_lines()
		self._beam_correct()
		
		return	
	
	def _set_arrays(self):
		if self.spectrum is None:
			self.spectrum = Spectrum()
		if self.source is None:
			self.source = Source()
		if isinstance(self.ll,list) is False:
			if isinstance(self.ll,np.ndarray):
				self.ll = self.ll.tolist()
			else:
				self.ll = [self.ll]
		if isinstance(self.ul,list) is False:
			if isinstance(self.ul,np.ndarray):
				self.ul = self.ul.tolist()
			else:
				self.ul = [self.ul]		
		self.spectrum.frequency,self.ll_idxs,self.ul_idxs = _trim_arr(self.mol.catalog.frequency,self.ll,self.ul,return_idxs=True)
		self.spectrum.freq0 = _trim_arr(self.mol.catalog.frequency,self.ll,self.ul,ll_idxs=self.ll_idxs,ul_idxs=self.ul_idxs)
		self.aij = _trim_arr(self.mol.catalog.aij,self.ll,self.ul,ll_idxs=self.ll_idxs,ul_idxs=self.ul_idxs)
		self.gup = _trim_arr(self.mol.catalog.gup,self.ll,self.ul,ll_idxs=self.ll_idxs,ul_idxs=self.ul_idxs)
		self.eup = _trim_arr(self.mol.catalog.eup,self.ll,self.ul,ll_idxs=self.ll_idxs,ul_idxs=self.ul_idxs)
		return
		
	def _apply_voffset(self):
		self.spectrum.frequency = _apply_vlsr(self.spectrum.freq0,self.source.velocity)
		return	
		
	def _calc_tau(self):
		self.spectrum.tau = ((self.aij * cm**3 * (self.source.column * 100**2) * 
								self.gup * (np.exp(-self.eup/self.source.Tex)) *
							 	(np.exp(h*self.spectrum.frequency*1E6/(k*self.source.Tex))-1)
							 )
							/
							(8*np.pi*(self.spectrum.frequency*1E6)**3 *
								self.source.dV*1000 * self.mol.q(self.source.Tex)
							)
					)
		return
		
	def _calc_bg(self):
		self.spectrum.Ibg = self.source.continuum.Ibg(self.spectrum.frequency)
		self.spectrum.Tbg = self.source.continuum.Tbg(self.spectrum.frequency)
		return
		
	def _calc_Iv(self):
		self.spectrum.Iv = ((2*h*self.spectrum.tau*(self.spectrum.frequency*1E6)**3)/
							cm**2 * (np.exp(h*self.spectrum.frequency*1E6 /
											(k*self.source.Tex)) -1 )
							)*1E26
		return

	def _calc_Tb(self,freq,tau,Tbg):
		'''
		Eq. A1 of Turner 1991.  Inline definition of J_T and J_Tbg have a typo - the extra
		'-1' at the end should be an exponential.  These should be in Planck.
		'''
		
		J_T = ((h*freq*10**6/k)*
			  (np.exp(((h*freq*10**6)/
			  (k*self.source.Tex))) -1)**-1
			  )
		J_Tbg = ((h*freq*10**6/k)*
			  (np.exp(((h*freq*10**6)/
			  (k*Tbg))) -1)**-1
			  )			  
		return (J_T - J_Tbg)*(1 - np.exp(-tau))
		
	def _beam_correct(self):
		if self.observation is not None:
			if self.observation.observatory.sd is True:
				self.spectrum.Tb,self.beam_dilution = _apply_beam(self.spectrum.frequency,self.spectrum.Tb,self.source.size,self.observation.observatory.dish,return_beam=True)
				self.spectrum.int_profile = _apply_beam(self.spectrum.freq_profile,self.spectrum.int_profile,self.source.size,self.observation.observatory.dish,return_beam=False)
		return	
		
	def _make_lines(self):
		if self.line_profile is None:
			return
		if self.line_profile.lower() in ['gaussian','gauss']:
			lls_raw = self.spectrum.frequency - self.sim_width*self.source.dV*self.spectrum.frequency/ckm
			uls_raw = self.spectrum.frequency + self.sim_width*self.source.dV*self.spectrum.frequency/ckm
			ll_trim = [lls_raw[0]]
			ul_trim = [uls_raw[0]]
			for ll,ul in zip(lls_raw[1:],uls_raw[1:]):
				if ll < ul_trim[-1]:
					ul_trim[-1] = ul
				else:
					ll_trim.append(ll)
					ul_trim.append(ul)
			ll_trim = np.array(ll_trim)
			ul_trim = np.array(ul_trim)		
			freq_arr = np.concatenate([np.arange(ll,ul,self.res) for ll,ul in zip(ll_trim,ul_trim)])
			tau_arr = np.zeros_like(freq_arr)
			l_idxs = [find_nearest(freq_arr,x) for x in lls_raw]
			u_idxs = [find_nearest(freq_arr,x) for x in uls_raw]		
			for x,y,ll,ul in zip(self.spectrum.frequency,self.spectrum.tau,l_idxs,u_idxs):
				tau_arr[ll:ul] += _make_gauss(x,y,freq_arr[ll:ul],self.source.dV,ckm)
			self.spectrum.tau_profile = tau_arr
			self.spectrum.freq_profile = freq_arr
			self.spectrum.Tbg_profile = self.source.continuum.Tbg(freq_arr)
			self.spectrum.int_profile = self._calc_Tb(freq_arr,tau_arr,self.spectrum.Tbg_profile)
			return
			
	def get_beam(self,freq):
		return 	206265 * 1.22 * ((freq*u.MHz).to(u.m, equivalencies=u.spectral()).value) / self.observation.observatory.dish		
				
	def update(self):
		self._set_arrays()
		self._apply_voffset()
		self._calc_tau()
		self._calc_bg()
		self._calc_Iv()
		self.spectrum.Tb = self._calc_Tb(self.spectrum.frequency,self.spectrum.tau,self.spectrum.Tbg)
		self._make_lines()
		self._beam_correct()
		return
