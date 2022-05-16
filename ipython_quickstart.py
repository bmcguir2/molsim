import sys, os
sys.path.insert(0, '/Users/Brett/Dropbox/Programs')

import numpy as np
from molsim.file_handling import load_mol, load_obs, load_multi_obs
from molsim.file_io import _write_xy
from molsim.constants import ccm, cm, ckm, h, k, kcm
from molsim.fitting import do_lsf
from molsim.classes import Workspace, Catalog, Transition, Level, Molecule, PartitionFunction, Spectrum, Simulation, Continuum, Source, Observatory, Observation, Iplot, Trace
from molsim.stats import get_rms
from molsim.utils import _trim_arr, find_nearest, _make_gauss, _apply_vlsr, find_limits, find_peaks, _get_res, _find_nans, _find_limit_idx, generate_spcat_qrots, process_mcmc_json
from molsim.functions import sum_spectra, velocity_stack, matched_filter, convert_spcat, resample_obs
from molsim.plotting import plot_mf, plot_stack, plot_sim, plot_html
from molsim.analysis import set_upper_limit
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy import signal
import json


#filepath = resource_filename(__name__,'tests/')