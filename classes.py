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
		
		
		
		
		
		
		
		
		
		
		
		