import numpy as np
from molsim.constants import ccm, cm, ckm, h, k, kcm

def _read_txt(filein):
	'''Reads in any txt file and returns a line by line array'''
	
	return_arr = []
	
	with open(filein, 'r') as input:
		for line in input:
			return_arr.append(line)

	return return_arr
	
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
					'qn1'		:	qn1,
					'qn2'		:	qn2,
					'qn3'		:	qn3,
					'qn4'		:	qn4,
					'qn5'		:	qn5,
					'qn6'		:	qn6,
					'qn7'		:	qn7,
					'qn8'		:	qn8,
					'qn9'		:	qn9,
					'qn10'		:	qn10,
					'qn11'		:	qn11,
					'qn12'		:	qn12,
				}
	
	return split_cat
		