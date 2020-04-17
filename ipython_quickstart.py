import numpy as np
from molsim import file_handling as fh
from molsim import tests as tests
from molsim.constants import ccm, cm, ckm, h, k, kcm
from pkg_resources import resource_filename
from molsim.classes import workspace, catalog

filepath = resource_filename(__name__,'tests/')