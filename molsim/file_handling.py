import numpy as np
import re
from numba import njit
from molsim.constants import ccm, cm, ckm, h, k, kcm
from molsim.classes import Workspace, Catalog, Transition, Level, Molecule, PartitionFunction, Continuum, Simulation, Spectrum, Observation
from molsim.utils import _trim_arr, find_nearest, _make_gauss, _make_qnstr, _make_level_dict
from molsim.stats import get_rms
from molsim.file_io import _read_txt, _read_xy, _write_xy
import math


def _read_txt(filein):
	'''Reads in any txt file and returns a line by line array'''
	
	return_arr = []
	
	with open(filein, 'r') as input:
		for line in input:
			return_arr.append(line)

	return return_arr


def _read_xy(filein):
	'''Reads in a two column x y file and returns the numpy arrays	'''
	
	x = []
	y = []
	
	with open(filein, 'r') as input:
		for line in input:
			x.append(np.float(line.split()[0].strip()))
			y.append(np.float(line.split()[1].strip()))
	
	x = np.asarray(x)
	y = np.asarray(y)
			
	return x,y		


def _read_spcat(filein):
	'''
	Reads an SPCAT catalog and returns spliced out numpy arrays.
	Catalog energy units for elow are converted to K from cm-1.	
	'''
	
	#read in the catalog
	raw_arr = _read_txt(filein)
	
	#set up some basic lists to populate
	frequency = []
	freq_err = []
	logint = []
	dof = []
	elow = []
	gup = []
	tag = []
	qnformat = []
	qn1 = []
	qn2 = []
	qn3 = []
	qn4 = []
	qn5 = []
	qn6 = []
	qn7 = []
	qn8 = []
	qn9 = []
	qn10 = []
	qn11 = []
	qn12 = []
	
	#split everything out	
	for x in raw_arr:
		frequency.append(x[:13].strip())
		freq_err.append(x[13:21].strip())
		logint.append(x[21:29].strip())
		dof.append(x[29:31].strip())
		elow.append(x[31:41].strip())
		gup.append(x[41:44].strip())
		tag.append(x[44:51].strip())
		qnformat.append(x[51:55].strip())
		qn1.append(x[55:57].strip())
		qn2.append(x[57:59].strip())
		qn3.append(x[59:61].strip())
		qn4.append(x[61:63].strip())
		qn5.append(x[63:65].strip())
		qn6.append(x[65:67].strip())
		qn7.append(x[67:69].strip())
		qn8.append(x[69:71].strip())
		qn9.append(x[71:73].strip())
		qn10.append(x[73:75].strip())
		qn11.append(x[75:77].strip())
		qn12.append(x[77:].strip())
		
	#now go through and fix everything into the appropriate formats and make numpy arrays as needed
	
	#we start with the easy ones that don't have nonsense letters
	frequency = np.array(frequency)
	frequency = frequency.astype(np.float64)
	freq_err = np.array(freq_err)
	freq_err = freq_err.astype(np.float64)	
	logint = np.array(logint)
	logint = logint.astype(np.float64)	
	dof = np.array(dof)
	dof = dof.astype(np.int)
	elow = np.array(elow)
	elow = elow.astype(np.float64)
	tag = np.array(tag)
	tag = tag.astype(np.int)
	qnformat = np.array(qnformat)
	qnformat = qnformat.astype(np.int)
	
	#convert elow to Kelvin
		
	elow /= kcm
	
	#now we use a sub-function to fix the letters and +/- that show up in gup, and the qns, and returns floats	
	def _fix_spcat(x):
		'''Fixes letters and +/- in something that's read in and returns floats'''
		
		#fix blanks - we just want them to be nice nones rather than empty strings
		
		if x == '':
			return None
		elif x == '+':
			return 1
		elif x == '-':
			return -1	
			
		sub_dict = {'a' : 100,
					'b' : 110,
					'c' : 120,
					'd' : 130,
					'e' : 140,
					'f' : 150,
					'g' : 160,
					'h' : 170,
					'i' : 180,
					'j' : 190,
					'k' : 200,
					'l' : 210,
					'm' : 220,
					'n' : 230,
					'o' : 240,
					'p' : 250,
					'q' : 260,
					'r' : 270,
					's' : 280,
					't' : 290,
					'u' : 300,
					'v' : 310,
					'w' : 320,
					'x' : 330,
					'y' : 340,
					'z' : 350,
					}
					
		alpha = re.sub('[^a-zA-Z]+','',x)
		
		if alpha == '':
			return int(x)
		else:		
			return sub_dict.get(alpha.lower(), 0) + int(x[1])
	
	#run the other arrays through the fixer, then convert them to what they need to be
	
	gup = [_fix_spcat(x) for x in gup]
	gup = np.array(gup)
	gup = gup.astype(int)
	qn1 = [_fix_spcat(x) for x in qn1]
	qn1 = np.array(qn1)
	qn1 = qn1.astype(int) if all(y is not None for y in qn1) is True else qn1
	qn2 = [_fix_spcat(x) for x in qn2]
	qn2 = np.array(qn2)
	qn2 = qn2.astype(int) if all(x is not None for x in qn2) is True else qn2			
	qn3 = [_fix_spcat(x) for x in qn3]
	qn3 = np.array(qn3)
	qn3 = qn3.astype(int) if all(x is not None for x in qn3) is True else qn3	
	qn4 = [_fix_spcat(x) for x in qn4]
	qn4 = np.array(qn4)
	qn4 = qn4.astype(int) if all(x is not None for x in qn4) is True else qn4	
	qn5 = [_fix_spcat(x) for x in qn5]
	qn5 = np.array(qn5)
	qn5 = qn5.astype(int) if all(x is not None for x in qn5) is True else qn5	
	qn6 = [_fix_spcat(x) for x in qn6]
	qn6 = np.array(qn6)
	qn6 = qn6.astype(int) if all(x is not None for x in qn6) is True else qn6	
	qn7 = [_fix_spcat(x) for x in qn7]
	qn7 = np.array(qn7)
	qn7 = qn7.astype(int) if all(x is not None for x in qn7) is True else qn7	
	qn8 = [_fix_spcat(x) for x in qn8]
	qn8 = np.array(qn8)
	qn8 = qn8.astype(int) if all(x is not None for x in qn8) is True else qn8	
	qn9 = [_fix_spcat(x) for x in qn9]
	qn9 = np.array(qn9)
	qn9 = qn9.astype(int) if all(x is not None for x in qn9) is True else qn9	
	qn10 = [_fix_spcat(x) for x in qn10]
	qn10 = np.array(qn10)
	qn10 = qn10.astype(int) if all(x is not None for x in qn10) is True else qn10	
	qn11 = [_fix_spcat(x) for x in qn11]
	qn11 = np.array(qn11)
	qn11 = qn11.astype(int) if all(x is not None for x in qn11) is True else qn11	
	qn12 = [_fix_spcat(x) for x in qn12]
	qn12 = np.array(qn12)
	qn12 = qn12.astype(int) if all(x is not None for x in qn12) is True else qn12	
	
	#make the qnstrings
	qn_list_up = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,None,None) for qn1,qn2,qn3,qn4,qn5,qn6 in zip(qn1,qn2,qn3,qn4,qn5,qn6)]
	qn_list_low = [_make_qnstr(qn7,qn8,qn9,qn10,qn11,qn12,None,None) for qn7,qn8,qn9,qn10,qn11,qn12 in zip(qn7,qn8,qn9,qn10,qn11,qn12)]
	
	
	split_cat = {
					'frequency'	: 	frequency,
					'freq_err'	:	freq_err,
					'logint'	:	logint,
					'dof'		:	dof,
					'elow'		:	elow,
					'gup'		:	gup,
					'tag'		:	tag,
					'qnformat'	:	qnformat,
					'qn1up'		:	qn1,
					'qn2up'		:	qn2,
					'qn3up'		:	qn3,
					'qn4up'		:	qn4,
					'qn5up'		:	qn5,
					'qn6up'		:	qn6,
					'qn7up'		:	np.full(len(frequency),None),
					'qn8up'		:	np.full(len(frequency),None),
					'qnup_str'	:	np.array(qn_list_up),
					'qn1low'	:	qn7,
					'qn2low'	:	qn8,
					'qn3low'	:	qn9,
					'qn4low'	:	qn10,
					'qn5low'	:	qn11,
					'qn6low'	:	qn12,
					'qn7low'	:	np.full(len(frequency),None),
					'qn8low'	:	np.full(len(frequency),None),
					'qnlow_str'	:	np.array(qn_list_low),
					'notes'		:	'Loaded from file {}' .format(filein)
				}
	
	return split_cat

def _read_spectrum(filein):
	'''
	Reads in an npz saved spectrum.  Returns a spectrum object.
	'''
	
	npz_dict = np.load(filein,allow_pickle=True)
	new_dict = {}
	for x in npz_dict:
		new_dict[x] = npz_dict[x]
	#sort some back to strings from numpy arrays
	entries = ['name','notes']
	for entry in entries:
		if entry in new_dict:
			new_dict[entry] = str(new_dict[entry])
	
	spectrum = Spectrum(freq0 = new_dict['freq0'],
						frequency = new_dict['frequency'],
						Tb = new_dict['Tb'],
						Iv = new_dict['Iv'],
						Tbg = new_dict['Tbg'],
						Ibg = new_dict['Ibg'],
						tau = new_dict['tau'],
						tau_profile = new_dict['tau_profile'],
						freq_profile = new_dict['freq_profile'],
						int_profile = new_dict['int_profile'],
						Tbg_profile = new_dict['Tbg_profile'],
						velocity = new_dict['velocity'],
						int_sim = new_dict['int_sim'],
						freq_sim = new_dict['freq_sim'],
						snr = new_dict['snr'],
						id = new_dict['id'],
						notes = new_dict['notes'],
						name = new_dict['name']
						)
						
	return spectrum								

def _load_catalog(filein,type='SPCAT',catdict=None):
	'''
	Reads in a catalog file of the specified type and returns a catalog object.  
	Optionally accepts a catdict dictionary to preload the catalog object with 
	additional information. Defaults to loading an spcat catalog.
	
	Anything in catdict will overwrite what's loaded in from the read catalog
	function, so use cautiously.
	'''

	if type.lower() == 'molsim':
		npz_dict = np.load(filein,allow_pickle=True)	
		new_dict = {}
		for x in npz_dict:
			new_dict[x] = npz_dict[x]
		#sort some back to strings from numpy arrays
		entries = ['version','source','last_update','contributor_name','contributor_email','notes','refs']
		for entry in entries:
			if entry in new_dict:
				new_dict[entry] = str(new_dict[entry])

	elif type.lower() == 'spcat':
		new_dict = _read_spcat(filein) #read in the catalog file and produce the
									   #dictionary
				
	elif type.lower() == 'freq_int':
		freq_tmp,int_tmp = 	_read_xy(filein) #read in a frequency intensity 
												   #delimited file
		new_dict = {}
		new_dict['frequency'] = freq_tmp
		new_dict['man_int'] = int_tmp
	
	if catdict is not None:
		for x in catdict:
			new_dict[x] = catdict[x] #either add it to the new_dict or overwrite it			
		
	cat = Catalog(catdict=new_dict) #make the catalog and return it
	
	return cat
			
def load_mol(filein,type='molsim',catdict=None,id=None,name=None,formula=None,
				elements=None,mass=None,A=None,B=None,C=None,muA=None,muB=None,
				muC=None,mu=None,Q=None,qnstrfmt=None,partition_dict=None,
				qpart_file=None):

	'''
	Loads a molecule in from a catalog file.  Default catalog type is molsim.  Override
	things with catdict.  Generates energy level objects, transition objects, a partition
	function object and	a molecule object which it returns.
	'''
	
	#load the catalog in
	cat = _load_catalog(filein,type=type,catdict=catdict)

	#now we have to make a hash for every entries upper and lower state.  If this already
	#exists in the catalog, great.  If not, we have to make it.
	if cat.qnup_str is None:
		qnups = [cat.qn1up,cat.qn2up,cat.qn3up,cat.qn4up,cat.qn5up,cat.qn6up,cat.qn7up,
					cat.qn8up]	
		qn_list_up = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8) for qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8 in zip(cat.qn1up,cat.qn2up,cat.qn3up,cat.qn4up,cat.qn5up,cat.qn6up,cat.qn7up,cat.qn8up)]
	else:
		qn_list_up = cat.qnup_str
		
	if cat.qnlow_str is None:
		qnlows = [cat.qn1low,cat.qn2low,cat.qn3low,cat.qn4low,cat.qn5low,cat.qn6low,
				cat.qn7low,cat.qn8low]
		qn_list_low = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8) for qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8 in zip(cat.qn1low,cat.qn2low,cat.qn3low,cat.qn4low,cat.qn5low,cat.qn6low,cat.qn7low,cat.qn8low)]
	else:
		qn_list_low = cat.qnlow_str
		
	level_qns = np.concatenate((qn_list_low,qn_list_up),axis=0)
	level_qns = list(set(list(level_qns))) #get the unique ones
	level_dict = dict.fromkeys(level_qns)			
	
	#now we find unique energy levels.  We just get the dictionary of levels, since
	#that function is computationally intensive and we want to njit it.
	
	level_dict = _make_level_dict(
									cat.qn1low,
									cat.qn2low,
									cat.qn3low,
									cat.qn4low,
									cat.qn5low,
									cat.qn6low,
									cat.qn7low,
									cat.qn8low,
									cat.qn1up,
									cat.qn2up,
									cat.qn3up,
									cat.qn4up,
									cat.qn5up,
									cat.qn6up,
									cat.qn7up,
									cat.qn8up,
									cat.frequency,
									cat.elow,
									cat.gup,									
									qn_list_low,
									qn_list_up,
									level_qns,
									level_dict,
									qnstrfmt,
								)
								
	#load those levels into level objects and add to a list
	levels = []
	for x in level_dict:
		levels.append(Level(
							energy = level_dict[x]['energy'],
							g = level_dict[x]['g'],
							g_flag = level_dict[x]['g_flag'],
							qn1 = level_dict[x]['qn1'],
							qn2 = level_dict[x]['qn2'],
							qn3 = level_dict[x]['qn3'],
							qn4 = level_dict[x]['qn4'],
							qn5 = level_dict[x]['qn5'],
							qn6 = level_dict[x]['qn6'],
							qn7 = level_dict[x]['qn7'],
							qn8 = level_dict[x]['qn8'],
							qnstrfmt = level_dict[x]['qnstrfmt'],
							id = level_dict[x]['id']
							))
	levels.sort(key=lambda x: x.energy) #sort them so the lowest energy is first
	
	#we'll now update the catalog with some of the things we've calculated like eup and
	#glow unless they're already present
	if type != 'molsim':
		level_ids = np.array([x.id for i,x in np.ndenumerate(levels)])
		tmp_dict = {}
		for x in levels:
			tmp_dict[x.id] = [x.g,x.energy]
		if cat.glow is None:
			cat.glow = np.empty_like(cat.frequency)	
		if cat.eup is None:
			cat.eup = np.empty_like(cat.frequency)			
		for x in range(len(cat.frequency)):
			cat.glow[x] = tmp_dict[cat.qnlow_str[x]][0]
			cat.eup[x] = tmp_dict[cat.qnup_str[x]][1]
	
	#now we have to load the transitions in	and make transition objects	
		
	#make the molecule
	mol = Molecule(levels=levels,catalog=cat)
	
	#make a partition function object and assign it to the molecule
	#if there's no other info, assume we're state counting
	if partition_dict is None:
		partition_dict = {}
		partition_dict['mol'] = mol
	#if there's a qpart file specified, read that in	
	if qpart_file is not None:
		partition_dict['qpart_file'] = qpart_file
	#make the partition function object and assign it	
	mol.qpart = PartitionFunction(	
				qpart_file = partition_dict['qpart_file'] if 'qpart_file' in partition_dict else None,
				form = partition_dict['form'] if 'form' in partition_dict else None,
				params = partition_dict['params'] if 'params' in partition_dict else None,
				temps = partition_dict['temps'] if 'temps' in partition_dict else None,
				vals = partition_dict['vals'] if 'vals' in partition_dict else None,
				mol = partition_dict['mol'] if 'mol' in partition_dict else None,
				gs = partition_dict['gs'] if 'gs' in partition_dict else None,
				energies = partition_dict['energies'] if 'energies' in partition_dict else None,
				sigma = partition_dict['sigma'] if 'sigma' in partition_dict else 1.,
				vib_states = partition_dict['vib_states'] if 'vib_states' in partition_dict else None,
				vib_is_K = partition_dict['vib_is_K'] if 'vib_is_K' in partition_dict else None,
				notes = partition_dict['notes'] if 'notes' in partition_dict else None,
							)
	
	#set sijmu and aij						
	mol.catalog._set_sijmu_aij(mol.qpart)						
	
	return	mol	
	
def load_obs(filein=None,xunits='MHz',yunits='K',id=None,notes=None,spectrum_id=None,spectrum_notes=None,source_dict=None,source=None,continuum_dict=None,continuum=None,observatory_dict=None,observatory=None,type='molsim'):
	
	'''
	Reads in an observations file and initializes an observation object with the given attributes.
	'''
	
	#initialize an Observation object
	obs = Observation()
	
	#read in the data if there is any
	if filein is not None:
		if type == 'molsim':
			obs.spectrum = _read_spectrum(filein)
		else:
			x,y = _read_xy(filein)
			if xunits == 'GHz':
				x*=1000
				xunits = 'MHz'
			obs.spectrum.frequency = x
			if yunits == 'K':
				obs.spectrum.Tb = y
			elif yunits.lower() == 'jy/beam':
				obs.spectrum.Iv = y
	
	if id is not None:
		obs.id = id
	if spectrum_id is not None:
		obs.spectrum.id = spectrum_id
	if notes is not None:
		obs.notes = notes
	if spectrum_notes is not None:
		obs.spectrum.notes = spectrum_notes
		
	if source is not None:
		obs.source = source
	elif source_dict is not None:
		obs.source = Source(
								name = source_dict['name'] if 'name' in source_dict else None,
								coords = source_dict['coords'] if 'coords' in source_dict else None,
								velocity = source_dict['velocity'] if 'velocity' in source_dict else 0.,
								size = source_dict['size'] if 'size' in source_dict else 1E20,
								solid_angle = source_dict['solid_angle'] if 'solid_angle' in source_dict else None,
								column = source_dict['column'] if 'column' in source_dict else 1.E13,
								Tex = source_dict['Tex'] if 'Tex' in source_dict else 300,
								Tkin = source_dict['Tkin'] if 'Tkin' in source_dict else None,
								dV = source_dict['dV'] if 'dV' in source_dict else 3.,
								notes = source_dict['notes'] if 'notes' in source_dict else None,	
							)
							
	if continuum is not None:
		obs.source.continuum = continuum
	elif continuum_dict is not None:
		obs.source.continuum = Continuum(
											cont_file = continuum_dict['cont_file'] if 'cont_file' in continuum_dict else None,
											type = continuum_dict['type'] if 'type' in continuum_dict else 'thermal',
											params = continuum_dict['params'] if 'params' in continuum_dict else [2.7],
											freqs = continuum_dict['freqs'] if 'freqs' in continuum_dict else None,
											temps = continuum_dict['temps'] if 'temps' in continuum_dict else None,
											fluxes = continuum_dict['fluxes'] if 'fluxes' in continuum_dict else None,
											notes = continuum_dict['notes'] if 'notes' in continuum_dict else None,
										)
	if observatory is not None:
		obs.observatory = observatory									
	elif observatory_dict is not None:
		obs.observatory = Observatory(
										name = observatory_dict['name'] if 'name' in observatory_dict else None,
										id = observatory_dict['id'] if 'id' in observatory_dict else None,
										sd = observatory_dict['sd'] if 'sd' in observatory_dict else True,
										array = observatory_dict['array'] if 'array' in observatory_dict else False,
										dish = observatory_dict['dish'] if 'dish' in observatory_dict else 100.,
										synth_beam = observatory_dict['synth_beam'] if 'synth_beam' in observatory_dict else [1.,1.],
										loc = observatory_dict['loc'] if 'loc' in observatory_dict else None,
										eta = observatory_dict['eta'] if 'eta' in observatory_dict else None,
										eta_type = observatory_dict['eta_type'] if 'eta_type' in observatory_dict else 'Constant',
										eta_params = observatory_dict['eta_params'] if 'eta_params' in observatory_dict else [1.],
										atmo = observatory_dict['atmo'] if 'atmo' in observatory_dict else None,
									)															

	return obs		

