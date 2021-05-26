import numpy as np
from numba import njit
from molsim.constants import ccm, cm, ckm, h, k, kcm
from molsim.stats import get_rms
import math
import warnings
from scipy import stats, signal
import sys, os
import json
from ruamel.yaml import YAML
from glob import glob
from typing import List
from multiprocessing import Pool
from subprocess import Popen, PIPE, TimeoutExpired
from tempfile import NamedTemporaryFile
from shutil import copy2, which


def load_yaml(yml_path: str) -> dict:
    with open(yml_path, "r") as read_file:
        yaml = YAML(typ="safe")
        input_dict = yaml.load(read_file)
    return input_dict


def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx 

def _trim_arr(arr,lls,uls,key_arr=None,return_idxs=False,ll_idxs=None,ul_idxs=None,return_mask=False):
	'''
	Trims the input array to the limits specified.  Optionally, will get indices from 
	the key_arr for trimming instead.
	'''
	
	if ll_idxs is not None:
		return np.concatenate([arr[ll_idx:ul_idx] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)])
	# modified to set as False to begin with, and working with
	# booleans instead of numbers
	mask_arr = np.zeros_like(arr, dtype=bool)
	if key_arr is None:	
		for x,y in zip(lls,uls):
			mask_arr[(arr>x) & (arr<y)] = True
	else:
		for x,y in zip(lls,uls):
			mask_arr[(key_arr>x) & (key_arr<y)] = True
	if return_mask:
		return mask_arr
	if return_idxs is False:
		return arr[mask_arr]
	else:
		ll_idxs_out = _find_ones(mask_arr)[0]
		ul_idxs_out = _find_ones(mask_arr)[1]
		return arr[mask_arr],ll_idxs_out,ul_idxs_out
		
@njit
def _make_gauss(freq0,int0,freq,dV,ckm):
	return int0*np.exp(-((freq-freq0)**2/(2*((dV*freq0/ckm)/2.35482)**2)))

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
	
def _make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8):
	qn_list = [qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8]
	tmp_list = [str(x).zfill(2) for x in qn_list if x != None]
	return ''.join(tmp_list)	

def _apply_vlsr(frequency,vlsr):
	'''
	Applies a vlsr shift to a frequency array.  Frequency in [MHz], vlsr in [km/s]
	'''
	return frequency - vlsr*frequency/ckm

# JIT'd version of the above function; PyMC3 no like Numba
_njit_apply_vlsr = njit(_apply_vlsr)
	
def _apply_beam(freq_arr,int_arr,source_size,dish_size,return_beam=False):
	beam_size = 206265 * 1.22 * (cm/(freq_arr * 1E6)) / dish_size #get beam size in arcsec
	beam_dilution = source_size**2 / (beam_size**2 + source_size**2)
	if return_beam is False:
		return int_arr*beam_dilution
	else:
		return int_arr*beam_dilution,beam_dilution

def find_limits(freq_arr,spacing_tolerance=100,padding=0):
	'''
	Finds the limits of a set of data, including gaps over a width, determined by the
	spacing tolerance.  Optional padding to each side to allow user to change vlsr and get 
	the simulation within the right area.
	'''
	
	# if len(freq_arr) == 0:
	# 	print('The input array has no data.')
	# 	return

	# this algorithm compares each gap against the average spacing of spacing_tolerance nearby points
	# if the gap is larger than spacing_tolerance * average spacing in both directions, it is considered as the actual gap
	# simplifying it gives the following expressions

	t = spacing_tolerance
	center = freq_arr[t+1:-t] - freq_arr[t:-t-1]
	left = freq_arr[t:-t-1] - freq_arr[:-2*t-1]
	right = freq_arr[2*t+1:] - freq_arr[t+1:-t]

	gaps = np.concatenate(([-1], np.where(np.logical_and(center > left, center > right))[0] + t, [-1]))

	ll = freq_arr[gaps[:-1]+1]
	ul = freq_arr[gaps[1:]]

	# the original expressions require 1 addition, 1 multiplication and 1 division operations per element, i.e. 3N operations, plus memory allocation and deallocation of temporary array
	# the following use 1 multiplication per element plus 1 addition and 1 division, i.e. N+2 operations, without allocating temporary array
	ll *= 1 - padding/ckm
	ul *= 1 + padding/ckm
	
	return ll,ul
	
def _find_limit_idx(freq_arr,spacing_tolerance=100,padding=25):		
	'''
	Finds the indices of the limits of a set of data, including gaps over a width, 
	determined by the  spacing tolerance.  Adds padding to each side to allow user to 
	change vlsr and get the simulation within the right area.
	'''
	
	if len(freq_arr) == 0:
		print('The input array has no data.')
		return

	#first, calculate the most common data point spacing as the mode of the spacings
	#this won't be perfect if the data aren't uniformly sampled
	#get the differences
	diffs = np.diff(freq_arr)
	spacing = stats.mode(diffs)[0][0]
	
	gaps = np.where(abs(diffs) > spacing*spacing_tolerance)
	
	ll = np.concatenate((np.array([freq_arr[0]]),freq_arr[gaps[0][:]+1]))
	ul = np.concatenate((freq_arr[gaps[0][:]],np.array([freq_arr[-1]])))
	
	ll -= padding*ll/ckm
	ul += padding*ul/ckm
	
	ll = [find_nearest(freq_arr,x) for x in ll]
	ul = [find_nearest(freq_arr,x) for x in ul]
	
	return ll,ul
	
def _find_nans(arr):
	'''
	Find the start,[stop] indices where value is present in arr
	'''

	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	# here .view(np.int8) changes the np.equal output from a bool array to a 1/0 array
	new_arr = np.copy(arr)
	new_arr[~np.isnan(new_arr)] = 0
	new_arr[np.isnan(new_arr)] = 1
	iszero = np.concatenate(([0], np.equal(arr, 1).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	
	lls = [x[0] for x in ranges]
	uls = [x[1] for x in ranges]
	
	return lls,uls	
	
def _find_ones(arr):
	'''
	Find the start,[stop] indices where value is present in arr
	'''

	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	# here .view(np.int8) changes the np.equal output from a bool array to a 1/0 array
	new_arr = np.copy(arr)
	iszero = np.concatenate(([0], np.equal(arr, 1).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	
	lls = [x[0] for x in ranges]
	uls = [x[1] for x in ranges]
	
	return lls,uls		

def find_peaks(freq_arr,int_arr,res,min_sep,is_sim=False,sigma=3,kms=True):
	'''
	'''

	if kms is True:
		max_f = np.amax(freq_arr)
		min_f = np.amin(freq_arr)
		cfreq = (max_f + min_f)/2
		v_res = res*ckm/max_f #finest velocity spacing
		v_span = (max_f - min_f) * ckm/(cfreq) #total velocity range spanned, setting cfreq at v=0.
		v_samp = np.arange(-v_span/2,v_span/2+v_res,v_res) #create a uniformly spaced velocity array
		freq_new = v_samp*cfreq/ckm + cfreq #convert it back to frequency
		int_new = np.interp(freq_new,freq_arr,int_arr,left=0.,right=0.)
		chan_sep = min_sep/v_res
	else:
		freq_new = freq_arr
		int_new = int_arr
		chan_sep = min_sep/res
	
	indices = signal.find_peaks(int_new,distance=chan_sep)

	if kms is True:
		indices = [find_nearest(freq_arr,freq_new[x]) for x in indices[0]] #if we had to re-sample things
		
	if is_sim is True:
		return np.asarray(indices)
		
	rms = get_rms(int_arr)
	indices = [x for x in indices if int_arr[x]>sigma*rms ]
	
	return np.asarray(indices)
	
def _get_res(freq_arr):
	'''
	Return the resolution of an array (best guess).
	'''
	diffs = np.diff(freq_arr)
	return stats.mode(diffs)[0][0]
	
def _make_fmted_qnstr(qns,qnstr_fmt=None):	
	'''
	Given a qnstr_fmt formatter declaration, turns a set of quantum numbers into a
	human readable output.
	
	For example, for methanol with some conditions, we want the final output to look like:
	
	1(1)-A vt=0 for the upper state of the 834.28 transition that has catalog qns of "1 1 - 0," we would use:
	
	'/#1/(/#2/)/#3[+=+ A,-=- A,= E]/ vt=/#4/'
	
	'''
	
	#if a qnstr_fmt is not given, return a nicely cleaned up string
	if qnstr_fmt is None:
		qn_bits = [f'{x:>3}' for x in qns]
		qnstr = ' '
		return qnstr.join(qn_bits)
	
	#Clean up the formatting input a bit
	base_str = qnstr_fmt.split('/')
	
	if base_str[0] == '':		
		del base_str[0]			
	if base_str[-1] == '':
		del base_str[-1]
	
	#apply the formatting
	for x in range(len(base_str)):		
		if '#' in base_str[x] and '[' not in base_str[x]:
			base_str[x] = base_str[x].replace('#','')
			idx = int(base_str[x])				
			base_str[x] = str(qns[idx-1])
			
		if '#' in base_str[x] and '[' in base_str[x]:		
			conditions = base_str[x].split('[')[1].replace(']','').split(',')
			idx = int(base_str[x].split('[')[0].replace('#',''))			
			
			for y in range(len(conditions)):
				conditions[y] = conditions[y].split('=')	
			value = str(qns[idx-1])
							
			for y in range(len(conditions)):
				if conditions[y][0] == value:
					base_str[x] = str(conditions[y][1])				
	
	#make the string and return it				
	qnstr = ''		
	return qnstr.join(base_str)	

		
def generate_spcat_qrots(basename,fileout=None,add_temps=None,kmax=150):
	'''
	Runs SPCAT in the background and generates a bunch of new partition function data
	points to be used in a qpart file, which it outputs.  Can be quite slow.
	'''
	
	#define all the filenames to be used
	int_file = f'{basename}.int'
	int_bak_file = f'{basename}.int.bak'
	out_file = f'{basename}.out'
	qpart_file = f'{basename}.qpart'
	
	#set up a dictionary to hold the temperatures we want partition function values at
	#fill it with the default set, some new defaults, plus any additional from add_temps
	q_values = {}
	default_temps = [1., 3., 5., 7., 9.375, 12., 15., 18.75, 25., 37.5, 45., 55., 65., 
						75., 95., 115., 135., 150., 175., 200., 225., 250., 275., 300., 
						400., 500.,
					]
	if add_temps is not None:
		for temp in add_temps:
			default_temps.append(temp)
	for temp in default_temps:
		q_values[temp] = None
	
	#set up a list of the values SPCAT already calculates so we don't do extra work
	spcat_defaults = [9.375, 18.75, 37.5, 75., 150., 225., 300.]
		
	#now we run spcat at each of the temperatures it doesn't already do automatically.
	#first save a copy of the original int file as backup
	os.system(f'cp {int_file} {int_bak_file}')
	
	for temp in default_temps:
	
		#if its one of spcat's defaults, just move on, we'll get those automatically
		if temp in spcat_defaults:
			continue
	
		#first read in the int file and modify it to a new temperature
		raw_int = []
	
		with open(int_file, 'r') as input:
			for line in input:
				raw_int.append(line)
			
		'''
		an example int file second line looks like this:
		0  123  3103007   0   150   -10. -10.   40 300
		we need to modify:
			the 8th index (the temperature) 
			the 4th index, the max k, to make sure it's high enough for an accurate q
			the 7th index, max frequency to make sure it's low enough we aren't overloading the computational time
		'''
	
		line_split = raw_int[1].split()
		line_split[4] = f'{kmax}'
		line_split[7] = '20'
		line_split[8] = str(temp)
		raw_int[1] = '  '.join(line_split) + '\n'

		with open(int_file, 'w') as output:
			for line in raw_int:
				output.write(line)
				
		#run spcat
		os.system(f'spcat {int_file}')
		
		#open the out file and extract the partition function information
		raw_out = []
		
		with open(out_file, 'r') as input:
			for line in input:
				raw_out.append(line)
				
		'''
		The output file is not always formatted the same, depending on how many 
		quantum numbers and such were used.  So we need to look for a key line that says
		we're getting to the partition functions.  That line is:
		
		TEMPERATURE - Q(SPIN-ROT.) - log Q(SPIN-ROT.)
		
		So we'll find that index, then just parse everything in the following lines.
		Those are of the format
		
		    300.000   3103007.2692    6.4918
		    
		Where we want to just split it out and get the 0-index (temperature) and 1-index
		which is Q.    
		'''				
		
		start_i = 0
		
		for i in range(len(raw_out)):
			if 'TEMPERATURE - Q(SPIN-ROT.) - log Q(SPIN-ROT.)' in raw_out[i]:
				start_i = i+1
				
		for line in raw_out[start_i:]:
			temp = float(line.split()[0])
			qval = float(line.split()[1])			
			q_values[temp] = qval
			
	#now we reset the int file back to the original and delete the backup
	os.system(f'mv {int_bak_file} {int_file}')			
			
	#make lists from the dictionary so we can sort them
	temps_l = list(q_values.keys())
	temps_l.sort()
	qvals_l = [q_values[i] for i in temps_l]
	
	#now we make the output file
	with open(qpart_file, 'w') as output:
		output.write('#form : interpolation\n')
		for t,q in zip(temps_l,qvals_l):
			output.write(f'{t} {q}\n')
	
def process_mcmc_json(json_file, molecule, observation, ll=0, ul=float('inf'), line_profile='Gaussian', res=0.0014, stack_params = None, stack_plot_params = None, make_plots=True, return_json = False):

	from molsim.classes import Source, Simulation
	from molsim.functions import sum_spectra, velocity_stack, matched_filter
	from molsim.plotting import plot_stack, plot_mf
	
	with open(json_file) as input:
		json_dict = json.load(input)
		
	n_sources = len(json_dict['SourceSize']['mean'])
	
	sources = []
	
	for size,vlsr,col,tex,dv in zip(
		json_dict['SourceSize']['mean'],
		json_dict['VLSR']['mean'],
		json_dict['NCol']['mean'],
		json_dict['Tex']['mean'],
		json_dict['dV']['mean']):
		sources.append(Source(size=size,velocity=vlsr,column=col,Tex=tex,dV=dv))
		
	sims = [Simulation(mol=molecule,ll=ll,ul=ul,observation=observation,source=x,line_profile=line_profile,res=res) for x in sources]	
	sum1 = sum_spectra(sims)
	
	if make_plots is False:
		if return_json is True:
			return sources, sims, sum1, json_dict
		else:
			return sources, sims, sum1
			
	internal_stack_params = {'selection' : 'lines',
					'freq_arr' : observation.spectrum.frequency,
					'int_arr' : observation.spectrum.Tb,
					'freq_sim' : sum1.freq_profile,
					'int_sim' : sum1.int_profile,
					'res_inp' : res,
					'dV' : np.mean([x for x in json_dict['dV']['mean']]),
					'dV_ext' : 40,
					'vlsr' : np.mean([x for x in json_dict['VLSR']['mean']]),
					'vel_width' : 40,
					'v_res' : 0.02,
					'blank_lines' : True,
					'blank_keep_range' : [-5*np.mean([x for x in json_dict['dV']['mean']]),5*np.mean([x for x in json_dict['dV']['mean']])],
					'flag_lines' : False,
					'flag_sigma' : 5,
					}	
	
	if stack_params is not None:
		for x in stack_params:
			internal_stack_params[x] = stack_params[x]
		
	stack = velocity_stack(internal_stack_params)
	
	internal_stack_plot_params = {'xlimits' : [-10,10]}
	
	if stack_plot_params is not None:
		for	x in stack_plot_params:
			internal_stack_plot_params[x] = stack_plot_params[x]
		
	plot_stack(stack,params=internal_stack_plot_params)	
	
	mf = matched_filter(stack.velocity,
						stack.snr,
						stack.int_sim[find_nearest(stack.velocity,-2):find_nearest(stack.velocity,2)])
	plot_mf(mf)
	
	if return_json is True:
		return sources, sims, sum1, json_dict
	else:
		return sources, sims, sum1	


def read_spcat_out(out_path):
    with open(out_path) as read_file:
        data = read_file.readlines()
    for index, line in enumerate(data):
        if line.startswith("TEMPERATURE"):
            break
    # drop the last line as well, which is "sorted ..."
    data = data[(index + 1):]
    output = dict()
    for line in data:
        split_line = line.split()
        # the dictionary keys are not converted to float
        # so that we can merge them later
        output[split_line[0]] = float(split_line[1])
    return output


def spcat_out_qpart(name: str, out_path: str):
    q_dict = read_spcat_out(out_path)
    q_dict = {float(key): value for key, value in q_dict.items()}
    with open(f"{name}.qpart", "w+") as write_file:
        write_file.write("# form : interpolation\n")
        for T, Q in q_dict.items():
            write_file.write(f"{T:.1f} {Q:.4f}\n")


def run_temperature(basename: str, int_contents: str, temperature: float):
    """
    'Micro'function that runs SPCAT with a given base file name, the contents
    of a .int file as a single string, and a target temperature.

    This function will create a temporary file to run SPCAT and should
    clean up after itself. The partition function values are parsed from
    the standard output, so we don't actually care about the .out file.

    Returns a dictionary containing the partition function at every
    temperature simulated in this run (i.e. including the SPCAT defaults)
    """
    temp = NamedTemporaryFile().name
    copy2(f"{basename}.var", f"{temp}.var")
    with open(f"{temp}.int", "w+") as write_file:
        write_file.write(
            int_contents.format(T=temperature)
        )
    with Popen(["spcat", temp], stdout=PIPE) as proc:
        try:
            out, errs = proc.communicate(timeout=600)
        except TimeoutExpired:
            proc.kill()
            out, errs = None
        # parse the partition functions out of the standard output
        if out is not None:
            q_dict = read_spcat_stdout(out)
        else:
            q_dict = None
    # cleanup after work
    for file in glob(f"{temp}*"):
        os.remove(file)
    return q_dict


def read_spcat_stdout(stdout):
    """
    This function simply parses the standard output of an SPCAT
    call, and extracts the partition function at each temperature
    simulated.
    """
    data = stdout.decode('utf-8').split("\n")
    # skip all of the extra stuff and go straight to
    # start reading the partition function
    for index, line in enumerate(data):
        if line.startswith("TEMPERATURE"):
            break
    # drop the last line as well, which is "sorted ..."
    data = data[(index + 1):-2]
    output = dict()
    for line in data:
        split_line = line.split()
        # the dictionary keys are not converted to float
        # so that we can merge them later
        output[split_line[0]] = float(split_line[1])
    return output


def parallel_spcat_qrots(basename: str, *more_temps, nproc: int = 4):
    if not which("spcat"):
        raise FileNotFoundError("SPCAT not located in your PATH! Check `which spcat` in your command line.")
    with open(f"{basename}.int", "r") as read_file:
        int_contents = read_file.read()
    var_file = f"{basename}.var"
    # some default temperatures to start with in addition
    # to the SPCAT ones
    temps = [5., 10., 25., 50., 100., 150., 200.]
    # user can provide any number more of temperatures
    if more_temps:
        temps.extend(more_temps)
    final_dict = dict()
    # runs SPCAT asynchronously
    with Pool(processes=nproc) as pool:
        results = [pool.apply_async(run_temperature, args=(basename, int_contents, T)) for T in temps]
        for result in results:
            final_dict.update(**result.get())
    # convert the temperatures into floats
    final_dict = {float(T): Q for T, Q in final_dict.items()}
    # write the results to disk
    with open(f"{basename}.qpart", "w+") as write_file:
        write_file.write("#form : interpolation\n")
        for T in sorted(final_dict.keys()):
            Q = final_dict.get(T)
            write_file.write(f"{float(T):.1f} {Q:.4f}\n")
    print(f"Partition functions written to {basename}.qpart")
				
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
				
			