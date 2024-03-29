# molsim

## Simulation tools for radio molecular spectra

`molsim` is a Python 3 package that provides an object-oriented interface for analyzing molecular spectra in high resolution astronomical observations and laboratory data. Key features of `molsim` include:

1. Line profile simulation
2. Velocity stacking
3. Interface to `emcee` for MCMC analysis
4. Matched filter analysis

For details about the methodology, particularly with respect to MCMC simulations, please refer to [Loomis _et al._ 2020](https://arxiv.org/abs/2009.11900)

If you use `molsim` for your analysis, please cite the Zenodo entry: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8118192.svg)](https://doi.org/10.5281/zenodo.8118192)

## Setup instructions

For science-ready code, we recommend downloading one of the releases—these are verified to the best of our ability to be accurate. If there are any indications otherwise, please submit an issue!

For the latest build, either for testing or for contributing, please clone the `development` branch of this repository.

We recommend using `conda` for maintaining Python environments. Once you have acquired the code either from downloading a release or cloning the repository, you can create a new `conda` environment (called `molsim`) by running the following in the `molsim` directory:

`conda env create -n molsim -f conda.yml`

followed by:

`conda activate molsim`

to change the Anaconda environment, and then

`pip install .`

which will then install `molsim` into your Anaconda/Python installation.

For developers/testers, you should make a fork of this repository, and make changes to the `development` branch. To separate science/production and development environments, make a new `conda` environment with the following command:

`conda env create -n molsim-dev -f conda.yml`

followed by:

`pip install -e .\[dev\]`

The backslashes are required to escape the `[]` characters for `zsh`, although you may not have that issue on other shells/OS'. This will install `molsim` as a softlink so that changes are updated on the fly, while the `[dev]` option installs additional packages such as `pytest`, and `black` for formatting.

### Use with notebook environments

A common way to interact with notebook environments is to have a `base` environment installation of `jupyter`. To use the Python kernel installed as part of the `molsim` environment with the `base` environment, using these steps:

1. `conda activate molsim`
2. `python -m ipykernel install --user --name molsim --display-name "molsim-jupyter"`

This links the IPython kernel with the `jupyter` installation. You'll then be able to start a notebook with the environment called "molsim-jupyter".

## Contributions

The large majority of `molsim` was written by @bmcguir2, building on top of earlier code (`simulate_lte`).

The `mcmc` module was written by @laserkelvin, based heavily on earlier code by @ryanaloomis (see [his repo here](https://github.com/ryanaloomis/TMC1_mcmc_fitting)), which was used for the DR1 GOTHAM data analysis. The `radex` module was written by @cixue, implemented based on the RADEX non-LTE radiative transfer code, and accelerated with `Numba` to provide the high performance required for MCMC analysis.

Any issues, please submit an issue, reporting what you think should happen and what actually happens.
