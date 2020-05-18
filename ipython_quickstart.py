import sys
sys.path.insert(0, '/Users/Brett/Dropbox/Programs')

import numpy as np
from molsim import file_handling as fh
from molsim.file_handling import load_mol
from molsim.constants import ccm, cm, ckm, h, k, kcm
from pkg_resources import resource_filename
from molsim.classes import Workspace, Catalog, Transition, Level, Molecule, PartitionFunction, Spectrum, Simulation, Continuum
from molsim.stats import get_rms
from molsim.utils import _trim_arr, find_nearest
import matplotlib.pyplot as plt
import matplotlib
import math

filepath = resource_filename(__name__,'tests/')