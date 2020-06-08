import numpy as np

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
	
def _write_xy(x,y,fileout):
	'''Writes out a two column x y file, tab delimited'''
	
	with open(fileout,'w') as output:
		for x0,y0 in zip(x,y):
			output.write('{}\t{}\n' .format(x0,y0))
			
	return