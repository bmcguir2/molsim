import numpy as np
from numba import njit
from molsim.constants import ccm, cm, ckm, h, k, kcm
from molsim.classes import Workspace, Catalog, Transition, Level, Molecule, PartitionFunction


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
			
		#fix +/-: we'll just turn it into 1 or -1
		
		if x == '+':
			return 1
			
		if x == '-':
			return -1
		
		#and now the letters bullshit
			
		if 'A' in x:		
			return 100 + int(x[1])
		
		if 'B' in x:		
			return 110 + int(x[1])	
		
		if 'C' in x:		
			return 120 + int(x[1])		

		if 'D' in x:		
			return 130 + int(x[1])
		
		if 'E' in x:		
			return 140 + int(x[1])
		
		if 'F' in x:		
			return 150 + int(x[1])
		
		if 'G' in x:		
			return 160 + int(x[1])
		
		if 'H' in x:		
			return 170 + int(x[1])				
		
		if 'I' in x:		
			return 180 + int(x[1])	
		
		if 'J' in x:		
			return 190 + int(x[1])
		
		if 'K' in x:		
			return 200 + int(x[1])
		
		if 'L' in x:		
			return 210 + int(x[1])
		
		if 'M' in x:		
			return 220 + int(x[1])	
		
		if 'N' in x:		
			return 230 + int(x[1])	
		
		if 'O' in x:		
			return 240 + int(x[1])
		
		if 'P' in x:		
			return 250 + int(x[1])
		
		if 'Q' in x:		
			return 260 + int(x[1])	
		
		if 'R' in x:		
			return 270 + int(x[1])
		
		if 'S' in x:		
			return 280 + int(x[1])
		
		if 'T' in x:		
			return 290 + int(x[1])	
		
		if 'U' in x:		
			return 300 + int(x[1])	
		
		if 'V' in x:		
			return 310 + int(x[1])
		
		if 'W' in x:		
			return 320 + int(x[1])	
		
		if 'X' in x:		
			return 330 + int(x[1])	
		
		if 'Y' in x:		
			return 340 + int(x[1])	
		
		if 'Z' in x:		
			return 350 + int(x[1])
		
		if 'a' in x:		
			return 100 + int(x[1])
		
		if 'b' in x:		
			return 110 + int(x[1])	
		
		if 'c' in x:		
			return 120 + int(x[1])		

		if 'd' in x:		
			return 130 + int(x[1])
		
		if 'e' in x:		
			return 140 + int(x[1])
		
		if 'f' in x:		
			return 150 + int(x[1])
		
		if 'g' in x:		
			return 160 + int(x[1])
		
		if 'h' in x:		
			return 170 + int(x[1])				
		
		if 'i' in x:		
			return 180 + int(x[1])	
		
		if 'j' in x:		
			return 190 + int(x[1])
		
		if 'k' in x:		
			return 200 + int(x[1])
		
		if 'l' in x:		
			return 210 + int(x[1])
		
		if 'm' in x:		
			return 220 + int(x[1])	
		
		if 'n' in x:		
			return 230 + int(x[1])	
		
		if 'o' in x:		
			return 240 + int(x[1])
		
		if 'p' in x:		
			return 250 + int(x[1])
		
		if 'q' in x:		
			return 260 + int(x[1])	
		
		if 'r' in x:		
			return 270 + int(x[1])
		
		if 's' in x:		
			return 280 + int(x[1])
		
		if 't' in x:		
			return 290 + int(x[1])	
		
		if 'u' in x:		
			return 300 + int(x[1])	
		
		if 'v' in x:		
			return 310 + int(x[1])
		
		if 'w' in x:		
			return 320 + int(x[1])	
		
		if 'x' in x:		
			return 330 + int(x[1])	
		
		if 'y' in x:		
			return 340 + int(x[1])	
		
		if 'z' in x:		
			return 350 + int(x[1])
			
		return int(x)						
	
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
					'qn1low'	:	qn7,
					'qn2low'	:	qn8,
					'qn3low'	:	qn9,
					'qn4low'	:	qn10,
					'qn5low'	:	qn11,
					'qn6low'	:	qn12,
				}
	
	return split_cat


def _load_catalog(filein,type='SPCAT',catdict=None):
	'''
	Reads in a catalog file of the specified type and returns a catalog object.  
	Optionally accepts a catdict dictionary to preload the catalog object with 
	additional information. Defaults to loading an spcat catalog.
	
	Anything in catdict will overwrite what's loaded in from the read catalog
	function, so use cautiously.
	'''

	if type == 'SPCAT':
		new_dict = _read_spcat(filein) #read in the catalog file and produce the
									   #dictionary
		#prep a bunch of empty arrays
		new_arrs = ['glow','eup','sijmu','sij','aij','measured','types']
		for x in new_arrs:
			new_dict[x] = np.empty_like(new_dict['frequency'])
				
	if type == 'freq_int':
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
			
def _make_level_dict(qn1low,qn2low,qn3low,qn4low,qn5low,qn6low,qn7low,qn8low,qn1up,qn2up,
					qn3up,qn4up,qn5up,qn6up,qn7up,qn8up,frequency,elow,gup,
					qn_list_low,qn_list_up,level_qns,level_dict,qnstrfmt=None):

	#a list to hold levels
	levels = []
	
	#we need to sort out unique levels from our catalog.  Those will have unique quantum
	#numbers. When we find a match to a lower level, add the info in.
	for x in range(len(frequency)):
		qnstr_low = qn_list_low[x]
		level_dict[qnstr_low] = {'energy'	:	elow[x],
								 'g'		:	None,
								 'g_flag'	:	False,
								 'qn1'		:	qn1low[x] if qn1low is not None else None,
								 'qn2'		:	qn2low[x] if qn2low is not None else None,
								 'qn3'		:	qn3low[x] if qn3low is not None else None,
								 'qn4'		:	qn4low[x] if qn4low is not None else None,
								 'qn5'		:	qn5low[x] if qn5low is not None else None,
								 'qn6'		:	qn6low[x] if qn6low is not None else None,
								 'qn7'		:	qn7low[x] if qn7low is not None else None,
								 'qn8'		:	qn8low[x] if qn8low is not None else None,
								 'id'		:	qn_list_low[x],
								 'qnstrfmt'	:	qnstrfmt,			
								}
	
	#do it again to fill in energy levels that were upper states and didn't get hit
	for x in range(len(frequency)):
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up] is None:
			#calculate the energy.  Move the transition from MHz -> cm-1 -> K
			freq_cm = (frequency[x]*1E6/ccm)
			freq_K = freq_cm / kcm
			level_dict[qnstr_up] = {'energy'	:	elow[x] + freq_K,
									 'g'		:	gup[x],
									 'g_flag'	:	False,
									 'qn1'		:	qn1up[x] if qn1up is not None else None,
									 'qn2'		:	qn2up[x] if qn2up is not None else None,
									 'qn3'		:	qn3up[x] if qn3up is not None else None,
									 'qn4'		:	qn4up[x] if qn4up is not None else None,
									 'qn5'		:	qn5up[x] if qn5up is not None else None,
									 'qn6'		:	qn6up[x] if qn6up is not None else None,
									 'qn7'		:	qn7up[x] if qn7up is not None else None,
									 'qn8'		:	qn8up[x] if qn8up is not None else None,
									 'id'		:	qn_list_up[x],
									 'qnstrfmt'	:	qnstrfmt,			
									}			
	
	#go grab the degeneracies	
	for x in range(len(frequency)):	
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up]['g'] is None:
			level_dict[qnstr_up]['g'] = gup[x]
			
	#now go through and fill any degeneracies that didn't get hit (probably ground states)
	#assume it's just 2J+1.  Set the flag for a calculated degeneracy to True.
	for x in level_dict:
		if level_dict[x]['g'] is None:
			level_dict[x]['g'] = 2*level_dict[x]['qn1'] + 1	
			level_dict[x]['g_flag'] = True	

	return level_dict

def load_mol(filein,type='SPCAT',catdict=None,id=None,name=None,formula=None,
				elements=None,mass=None,A=None,B=None,C=None,muA=None,muB=None,
				muC=None,mu=None,Q=None,qnstrfmt=None,partition_dict=None,
				qpart_file=None):

	'''
	Loads a molecule in from a catalog file.  Default catalog type is SPCAT.  Override
	things with catdict.  Generates energy level objects, transition objects, a partition
	function object and	a molecule object which it returns.
	'''
	
	#load the catalog in
	cat = _load_catalog(filein,type=type,catdict=catdict)
	
	#get some qnstrings to use as hashes		
	#first we make subfunctions to generate qn_strs
	
	def _make_qnstr(y,qnlist):
		qn_list_trimmed = [x for x in qnlist if np.all(x) != None]
		tmp_list = [str(x[y]) for x in qn_list_trimmed]
		return ''.join(tmp_list)
	
	#now we have to make a hash for every entries upper and lower state
	qnlows = [cat.qn1low,cat.qn2low,cat.qn3low,cat.qn4low,cat.qn5low,cat.qn6low,
				cat.qn7low,cat.qn8low]
	qnups = [cat.qn1up,cat.qn2up,cat.qn3up,cat.qn4up,cat.qn5up,cat.qn6up,cat.qn7up,
				cat.qn8up]
	qn_list_low = [_make_qnstr(y,qnlows) for y in range(len(cat.frequency))]
	qn_list_up = [_make_qnstr(y,qnups) for y in range(len(cat.frequency))]
	level_qns = qn_list_low + qn_list_up
	level_qns = list(set(level_qns)) #get the unique ones
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
	#glow
	level_ids = np.array([x.id for i,x in np.ndenumerate(levels)])
	for x in range(len(cat.frequency)):
		line_qns_up_str = _make_qnstr(x,qnups)
		line_qns_low_str = _make_qnstr(x,qnlows)
		cat.glow[x] = levels[np.where(level_ids == line_qns_low_str)[0][0]].g
		cat.eup[x] = levels[np.where(level_ids == line_qns_up_str)[0][0]].energy
	
	#now we have to load the transitions in	and make transition objects	
		
	#make the molecule
	mol = Molecule(levels=levels,catalog=cat)
	
	#make a partition function object and assign it to the molecule
	#if there's no other info, assume we're state counting
	if partition_dict is None and qpart_file is None:
		partition_dict = {}
		partition_dict['mol'] = mol
	#if there's a qpart file specified, read that in	
	elif qpart_file is not None:
		temps,vals = _read_xy(qpart_file)
		if partition_dict is None:
			partition_dict = {}
		partition_dict['temps'] = temps
		partition_dict['vals'] = vals		
		partition_dict['notes'] = 'Loaded values from {}' .format(qpart_file)	
	#make the partition function object and assign it	
	mol.qpart = PartitionFunction(	
				form = partition_dict['form'] if 'form' in partition_dict else None,
				params = partition_dict['params'] if 'params' in partition_dict else None,
				temps = partition_dict['temps'] if 'temps' in partition_dict else None,
				vals = partition_dict['vals'] if 'vals' in partition_dict else None,
				mol = partition_dict['mol'] if 'mol' in partition_dict else None,
				gs = partition_dict['gs'] if 'gs' in partition_dict else None,
				energies = partition_dict['energies'] if 'energies' in partition_dict else None,
				sigma = partition_dict['sigma'] if 'sigma' in partition_dict else None,
				vib_states = partition_dict['vib_states'] if 'vib_states' in partition_dict else None,
				vib_is_K = partition_dict['vib_is_K'] if 'vib_is_K' in partition_dict else None,
				notes = partition_dict['notes'] if 'notes' in partition_dict else None,
							)
	
	return	mol	