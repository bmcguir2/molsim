class workspace(object):

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
		
		
class catalog(object):

	'''
	The catalog class holds all the spectroscopic data for a single molecule.
	''' 		
	
	def __init__(
					self,
					catdict = None, #if attributes are to be read in from a pre-made dictionary
					catid = None, #a unique catalog id
					molecule = None, #molecule name
					frequency = None, #frequencies [MHz]
					freq_err = None, #errors on frequencies [MHz]
					measured = None, #set to True if the line is known in the laboratory, False otherwise
					logint = None, #logarithmic intensities [log10(nm^2 MHz)]
					sijmu = None, #linestrength times the dipole moment squared [Debye^2]
					sij = None, #intrinsic linestrength (unitless)
					aij = None, #einstein A-coefficients [s^-1]
					man_int = None, #manually entered intensities, not to be used by calculations
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
					qn1low = None, #lower state principle quantum number 1
					qn2low = None, #lower state quantum number 2
					qn3low = None, #lower state quantum number 3
					qn4low = None, #lower state quantum number 4
					qn5low = None, #lower state quantum number 5
					qn6low = None, #lower state quantum number 6
					qn7low = None, #lower state quantum number 7
					qn8low = None, #lower state quantum number 8		
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
		self.man_it = man_int
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
		self.qn1low = qn1low
		self.qn2low = qn2low
		self.qn3low = qn3low
		self.qn4low = qn4low
		self.qn5low = qn5low
		self.qn6low = qn6low
		self.qn7low = qn7low
		self.qn8low = qn8low		
		self.version = version
		self.source = source
		self.last_update = last_update
		self.contributor_name = contributor_name
		self.contributor_email = contributor_email
		self.notes = notes
		self.refs = refs
		
		self.unpack_catdict()
	
		return
		
	def unpack_catdict(self):
	
		'''
		If a dictionary of data was provided, go ahead and use that to unpack things.
		'''
	
		if self.catdict is not None:	
			self.catid = self.catdict['catid'] if 'catid' in self.catdict and self.catid is None
			self.molecule = molecule
			self.frequency = frequency
			self.freq_err = freq_err
			self.measured = measured
			self.logint = logint
			self.sijmu = sijmu
			self.sij = sij
			self.aij = aij
			self.man_it = man_int
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
			self.qn1low = qn1low
			self.qn2low = qn2low
			self.qn3low = qn3low
			self.qn4low = qn4low
			self.qn5low = qn5low
			self.qn6low = qn6low
			self.qn7low = qn7low
			self.qn8low = qn8low		
			self.version = version
			self.source = source
			self.last_update = last_update
			self.contributor_name = contributor_name
			self.contributor_email = contributor_email
			self.notes = notes
			self.refs = refs
		
		return
		
		
		
		
		
		
		
		
		
		
		
		