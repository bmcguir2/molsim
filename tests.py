import numpy as np
import file_handling as fh
from pkg_resources import resource_filename

filepath = resource_filename(__name__,'tests/')

def _test_molsim():
	'''Meta function to run all validation and verification tests'''
	
	_val_molsim()
	print('\n')
	_ver_molsim()
	
	return

def _val_molsim():
	'''Meta function to run all validation test'''
	
	print('==========================')
	print('Beginning Validation Tests')
	print('==========================\n')
	
	_val_read_freq_int()
	_val_read_txt()	
	_val_read_spcat()
	
	return

def _ver_molsim():
	'''Meta function to run all verification test'''

	print('============================')
	print('Beginning Verification Tests')
	print('============================\n')	
	
	_ver_read_freq_int()
	_ver_read_txt()	
	_ver_read_spcat()
	
	return	

def _val_read_freq_int():

	'''Validate the reading in of a frequency intensity delimited file'''

	print('... validating file_handling._read_freq_int')

	try:
		frequency, intensity = fh._read_freq_int(filepath + 'file_handling/freq_int.txt')
	except:
		print('\tCRITICAL: file_handling._read_freq_int failed entirely to run.')
		return
		
	warnings = 0	
	
	#check that it has returned numpy arrays
	
	if isinstance(frequency,np.ndarray) is False:
		print('\tWARNING: file_handling._read_freq_int returned "frequency" as {}.  Expected np.ndarray.' .format(type(frequency)))
		warnings += 1
		
	if isinstance(intensity,np.ndarray) is False:
		print('\tWARNING: file_handling._read_freq_int returned "intensity" as {}.  Expected np.ndarray.' .format(type(intensity)))
		warnings += 1
		
	#check the length of the returned arrays
	
	if len(frequency) != 22:
		print('\tWARNING: file_handling._read_freq_int returned "frequency" with a length of {}.  Expected 22.' .format(len(frequency)))
		warnings += 1			
	
	if len(intensity) != 22:
		print('\tWARNING: file_handling._read_freq_int returned "intensity" with a length of {}.  Expected 22.' .format(len(intensity)))
		warnings += 1
		
	#check that the values are np.float64
	
	if any([isinstance(x,np.float64) for x in frequency]) is False:
		print('\tWARNING: file_handling._read_freq_int returned "frequency" with values that were not np.float64.')
		warnings += 1
		
	if any([isinstance(x,np.float64) for x in intensity]) is False:
		print('\tWARNING: file_handling._read_freq_int returned "intensity" with values that were not np.float64.')
		warnings += 1	
		
	print('\tValidation of file_handling._read_freq_int completed with {} warnings.\n' .format(warnings))						
	
	return warnings
	
def _val_read_txt():

	'''Validate the reading in of a generic text file into a line by line array'''
	
	print('... validating file_handling._read_txt')

	try:
		return_arr = fh._read_txt(filepath + 'file_handling/freq_int.txt')
	except:
		print('\tCRITICAL: file_handling._read_txt failed entirely to run.')
		return
		
	warnings = 0	
	
	#check that it has returned a list
	
	if isinstance(return_arr,list) is False:
		print('\tWARNING: file_handling._read_txt returned "return_arr" as {}.  Expected list.' .format(type(return_arr)))
		warnings += 1
		
	#check the length of the returned list
	
	if len(return_arr) != 22:
		print('\tWARNING: file_handling._read_txt returned "return_arr" with a length of {}.  Expected 22.' .format(len(return_arr)))
		warnings += 1			
		
	#check that the values are strings
	
	if any([isinstance(x,str) for x in return_arr]) is False:
		print('\tWARNING: file_handling._read_txt returned "return_arr" with values that were not strings.')
		warnings += 1	
		
	print('\tValidation of file_handling._read_txt completed with {} warnings.\n' .format(warnings))						
	
	return warnings
	
def _val_read_spcat():

	'''Validate the reading in of an spcat formatted cat file'''
	
	print('... validating file_handling._read_spcat')

	try:
		split_cat = fh._read_spcat(filepath + 'file_handling/testcat.cat')
	except:
		print('\tCRITICAL: file_handling._read_spcat failed entirely to run.')
		return
		
	warnings = 0
	
	frequency = split_cat['frequency']
	freq_err = split_cat['freq_err']
	logint = split_cat['logint']
	dof = split_cat['dof']
	elower = split_cat['elower']
	gup = split_cat['gup']
	tag = split_cat['tag']
	qnformat = split_cat['qnformat']
	qn1 = split_cat['qn1']
	qn2 = split_cat['qn2']
	qn3 = split_cat['qn3']
	qn4 = split_cat['qn4']
	qn5 = split_cat['qn5']
	qn6 = split_cat['qn6']
	qn7 = split_cat['qn7']
	qn8 = split_cat['qn8']
	qn9 = split_cat['qn9']
	qn10 = split_cat['qn10']
	qn11 = split_cat['qn11']
	qn12 = split_cat['qn12']
	
	floats = ['frequency', 'freq_err', 'logint', 'elower']
	ints = ['dof', 'gup', 'tag', 'qnformat', 'qn1', 'qn2', 'qn3', 'qn4', 'qn5', 'qn6', 'qn7', 'qn8', 'qn9', 'qn10', 'qn11', 'qn12']		
	
	#check that it has returned the appropriate types, with the appropriate lengths
	
	for x in split_cat:
		if isinstance(split_cat[x],np.ndarray) is False:
			print('\tWARNING: file_handling._read_spcat returned {} as {}.  Expected np.ndarray.' .format(x,type(split_cat[x])))
			warnings += 1
		if len(split_cat[x]) != 18:
			print('\tWARNING: file_handling._read_spcat returned {} with a length of {}.  Expected 18.' .format(x,len(split_cat[x])))
			warnings += 1			
		
	#check that the arrays that should be floats are floats
	
	for x in floats:
		if all(y is None for y in split_cat[x]) is True:
			continue
		if any([isinstance(y,np.float64) for y in split_cat[x]]) is False:
			print('\tWARNING: file_handling._read_spcat returned {} with values that were not np.float64 or None' .format(x))
			warnings += 1	
			
	#check that the arrays that should be ints are ints
	
	for x in ints:
		if all(y is None for y in split_cat[x]) is True:
			continue
		if any([isinstance(y,np.int64) for y in split_cat[x]]) is False:
			print('\tWARNING: file_handling._read_spcat returned {} with values that were not np.int64 or None' .format(x))
			warnings += 1		
		
	print('\tValidation of file_handling._read_spcat completed with {} warnings.\n' .format(warnings))						
	
	return warnings		
	
def _val_load_catalog():

	'''Validate the reading in of an spcat formatted cat file'''
	
	print('... validating file_handling._load_catalog')
	
	#We first try to load an spcat catalog
	
	try:
		cat = fh._load_catalog(filepath + 'file_handling/testcat.cat',type='SPCAT')
	except:
		print('\tCRITICAL: file_handling._load_cat failed entirely to run.')
		return			
		
	warnings = 0
	
	return warnings
	
def _ver_read_freq_int():

	'''Verify the reading in of a frequency intensity delimited file'''
	
	print('... verifying file_handling._read_freq_int')
	
	try:
		frequency, intensity = fh._read_freq_int(filepath + 'file_handling/freq_int.txt')
	except:
		print('\tCRITICAL: file_handling._read_freq_int failed entirely to run.')
		return

	warnings = 0	

	#check the total values
	
	if np.sum(frequency) != 29116838.174399998:
		print('\tWARNING: file_handling._read_freq_int returned "frequency" with a total value of {}.  Expected 29116838.174399998.' .format(np.sum(frequency)))
		warnings += 1
		
	if np.sum(intensity) != -62.5124:
		print('\tWARNING: file_handling._read_freq_int returned "intensity" with a total value of {}.  Expected 29116838.174399998.' .format(np.sum(intensity)))
		warnings += 1
		
	print('\tVerification of file_handling._read_freq_int completed with {} warnings.\n' .format(warnings))						
	
	return warnings	
	
def _ver_read_txt():

	'''Verify the reading in of a generic text file into a line by line array'''
	
	print('... verifying file_handling._read_txt')
	
	try:
		return_arr = fh._read_txt(filepath + 'file_handling/freq_int.txt')
	except:
		print('\tCRITICAL: file_handling._read_txt failed entirely to run.')
		return	
		
	warnings = 0	

	#check three values (start, middle, and end)

	if return_arr[0].strip() != '115271.2018 -5.0105':
		print('\tWARNING: file_handling._read_txt returned "return_arr" at index 0 of {}.  Expected 115271.2018 -5.0105' .format(return_arr[0].strip()))
		warnings += 1

	if return_arr[10].strip() != '1267014.4860 -2.3773':
		print('\tWARNING: file_handling._read_txt returned "return_arr" at index 0 of {}.  Expected 1267014.4860 -2.3773' .format(return_arr[0].strip()))
		warnings += 1
		
	if return_arr[-1].strip() != '2528172.0600 -2.9584':
		print('\tWARNING: file_handling._read_txt returned "return_arr" at index 0 of {}.  Expected 2528172.0600 -2.9584' .format(return_arr[0].strip()))
		warnings += 1		
		
	print('\tVerification of file_handling._read_txt completed with {} warnings.\n' .format(warnings))						
	
	return warnings

	
def _ver_read_spcat():

	'''Verify the reading in of an spcat formatted cat file'''
	
	print('... verifying file_handling._read_spcat')

	try:
		split_cat = fh._read_spcat(filepath + 'file_handling/testcat.cat')
	except:
		print('\tCRITICAL: file_handling._read_spcat failed entirely to run.')
		return
		
	warnings = 0
	
	frequency = split_cat['frequency']
	freq_err = split_cat['freq_err']
	logint = split_cat['logint']
	dof = split_cat['dof']
	elower = split_cat['elower']
	gup = split_cat['gup']
	tag = split_cat['tag']
	qnformat = split_cat['qnformat']
	qn1 = split_cat['qn1']
	qn2 = split_cat['qn2']
	qn3 = split_cat['qn3']
	qn4 = split_cat['qn4']
	qn5 = split_cat['qn5']
	qn6 = split_cat['qn6']
	qn7 = split_cat['qn7']
	qn8 = split_cat['qn8']
	qn9 = split_cat['qn9']
	qn10 = split_cat['qn10']
	qn11 = split_cat['qn11']
	qn12 = split_cat['qn12']	
	
	#check some sums as a first pass
	
	sums = {
				'frequency'	: 	23871.4071,
				'freq_err'	:	4.252800000000001,
				'logint'	:	-169.53010000000003,
				'dof'		:	54,
				'elower'	:	5831.896243295991,
				'gup'		:	5832,
				'tag'		:	1722,
				'qnformat'	:	5472,
				'qn1'		:	969,
				'qn2'		:	207,
				'qn3'		:	762,
				'qn4'		:	955,
				'qn5'		:	None,
				'qn6'		:	None,
				'qn7'		:	969,
				'qn8'		:	207,
				'qn9'		:	770,
				'qn10'		:	969,
				'qn11'		:	None,
				'qn12'		:	None,	
			}

	for x in sums:
		if sums[x] is None:
			continue #this was covered in validation
		if np.sum(split_cat[x]) != sums[x]:
			print('\tWARNING: file_handling._read_spcat returned the sum of {} as {}.  Expected {}.' .format(x,np.sum(split_cat[x]),sums[x]))
			warnings += 1
		
	print('\tVerification of file_handling._read_spcat completed with {} warnings.\n' .format(warnings))						
	
	return warnings	