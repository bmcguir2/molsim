import numpy as np
import itertools as it
import subprocess

def _expand_to_iterable(x):
    if type(x) is tuple:
        scale = x[0]
        low   = x[1]
        high  = x[2]
        num   = x[3]
        if scale == 'linear':
            return np.linspace(low, high, num=num, endpoint=True)
        if scale == 'log':
            return np.geomspace(low, high, num=num, endpoint=True)
        raise RuntimeError('Unregconized scale: ' + repr(scale))
    try:
        return iter(x)
    except TypeError:
        return [x]

def run(molfile='hco+.dat', outfile='radex.out', f_low=50, f_high=500, T_k=20.0, n_c={'H2': 1e4}, T_bg=2.73, N=1e13, dV=1.0):
    inp = []
    for f_low_, f_high_, T_bg_ in zip(*map(_expand_to_iterable, [f_low, f_high, T_bg])):
        for T_k_, N_, dV_ in it.product(*map(_expand_to_iterable, [T_k, N, dV])):
            for n_c_values in it.product(*map(_expand_to_iterable, n_c.values())):
                tmp = []
                tmp.append(molfile)
                tmp.append(outfile)
                tmp.append('%e %e' % (f_low_, f_high_))
                tmp.append('%e' % T_k_)
                tmp.append('%d' % len(n_c))
                for k, v in zip(n_c.keys(), n_c_values):
                    tmp.append(k)
                    tmp.append('%e' % v)
                tmp.append('%e' % T_bg_)
                tmp.append('%e' % N_)
                tmp.append('%e' % dV_)
                inp.append('\n'.join(tmp))
    inp = '\n1\n'.join(inp) + '\n0\n'
    subprocess.run(['radex'], input=inp, stdout=subprocess.DEVNULL, encoding='ascii')
    return outfile

def read(filename, squeeze=True):
    parameters_keys = []
    parameters_values = {}
    indices, results = [], []
    with open(filename, 'r') as f:
        # step = 0    : reading parameters
        # step = 1, 2 : dumping dummies
        # step = 3    : reading results
        step = 0
        index = []
        for line in f:
            if line[0] == '*':
                if step not in [0, 3]:
                    raise RuntimeError('Unexpected parameter line')
                if step == 3:
                    step = 0
                    index = []
                key, value = line.lstrip('*').split(':', 1)
                key = key.strip()
                if key == 'T(background)     [K]':
                    continue
                value = value.strip()
                try:
                    value = float(value)
                except:
                    pass
                if key not in parameters_keys:
                    parameters_keys.append(key)
                    parameters_values[key] = []
                try:
                    index.append(parameters_values[key].index(value))
                except ValueError:
                    index.append(len(parameters_values[key]))
                    parameters_values[key].append(value)
            else:
                if step == 0:
                    index.append(-1)
                    if 'LINE' not in parameters_keys:
                        parameters_keys.append('LINE')
                        parameters_values['LINE'] = []
                if step != 3:
                    step = step + 1
                    continue
                l = line.split()
                transition = ' '.join(l[0:3])
                try:
                    index[-1] = parameters_values['LINE'].index(transition)
                except ValueError:
                    index[-1] = len(parameters_values['LINE'])
                    parameters_values['LINE'].append(transition)
                indices.append(tuple(index))
                tmp = {}
                tmp['E_UP']     = float(l[3])
                tmp['FREQ']     = float(l[4])
                tmp['WAVEL']    = float(l[5])
                tmp['T_EX']     = float(l[6])
                tmp['TAU']      = float(l[7])
                tmp['T_R']      = float(l[8])
                tmp['POP_UP']   = float(l[9])
                tmp['POP_LOW']  = float(l[10])
                tmp['FLUX_K']   = float(l[11])
                tmp['FLUX_cgs'] = float(l[12])
                results.append(tmp)
    shape = [len(parameters_values[key]) for key in parameters_keys]
    grid = np.zeros(shape, dtype=object)
    for index, result in zip(indices, results):
        grid[index] = result
    if squeeze:
        for i, key in list(enumerate(parameters_keys)):
            if shape[i] == 1:
                parameters_keys.remove(key)
                del parameters_values[key]
        grid = np.squeeze(grid)
    return parameters_keys, parameters_values, grid
