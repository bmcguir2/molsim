import numpy as np
import file_handling as fh
from pkg_resources import resource_filename

filepath = resource_filename(__name__,'tests/')

def _val_read_freq_int():

	'''Validate the reading in of a frequency intensity delimited file'''

	try:
		frequency, intensity = fh._read_freq_int(filepath + 'file_handling/freq_int.txt')
	except:
		print('CRITICAL: file_handling._read_freq_int failed entirely to run.')
		return
		
	warnings = 0	
	
	#check that it has returned numpy arrays
	
	if isinstance(frequency,np.ndarray) is False:
		print('WARNING: file_handling._read_freq_int returned "frequency" as {}.  Expected np.ndarray.' .format(type(frequency)))
		warnings += 1
		
	if isinstance(intensity,np.ndarray) is False:
		print('WARNING: file_handling._read_freq_int returned "intensity" as {}.  Expected np.ndarray.' .format(type(intensity)))
		warnings += 1
		
	#check the length of the returned arrays
	
	if len(frequency) != 22:
		print('WARNING: file_handling._read_freq_int returned "frequency" with a length of {}.  Expected 22.' .format(len(frequency)))
		warnings += 1			
	
	if len(intensity) != 22:
		print('WARNING: file_handling._read_freq_int returned "intensity" with a length of {}.  Expected 22.' .format(len(intensity)))
		warnings += 1
		
	#check that the values are np.float64
	
	if any([isinstance(x,np.float64) for x in frequency]) is False:
		print('WARNING: file_handling._read_freq_int returned "frequency" with values that were not np.float64.')
		warnings += 1
		
	if any([isinstance(x,np.float64) for x in intensity]) is False:
		print('WARNING: file_handling._read_freq_int returned "intensity" with values that were not np.float64.')
		warnings += 1	
		
	print('Validation of file_handling._read_freq_int completed with {} warnings.' .format(warnings))						
	
	return
	
def _ver_read_freq_int():

	'''Verify the reading in of a frequency intensity delimited file'''

	try:
		frequency, intensity = fh._read_freq_int(filepath + 'file_handling/freq_int.txt')
	except:
		print('CRITICAL: file_handling._read_freq_int failed entirely to run.')
		return
		
	warnings = 0	

	#check the total values
	
	if np.sum(frequency) != 29116838.174399998:
		print('WARNING: file_handling._read_freq_int returned "frequency" with a total value of {}.  Expected 29116838.174399998.' .format(np.sum(frequency)))
		warnings += 1
		
	if np.sum(intensity) != -62.5124:
		print('WARNING: file_handling._read_freq_int returned "intensity" with a total value of {}.  Expected 29116838.174399998.' .format(np.sum(intensity)))
		warnings += 1
		
	print('Verification of file_handling._read_freq_int completed with {} warnings.' .format(warnings))						
	
	return
	
