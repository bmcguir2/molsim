# molsim

## Simulation tools for radio molecular spectra

`molsim` is a Python 3 package that provides an object-oriented interface for analyzing molecular spectra in high resolution astronomical observations and laboratory data. Key features of `molsim` include:

1. Line profile simulation
2. Velocity stacking
3. Interface to `emcee` for MCMC analysis
4. Matched filter analysis

For details about the methodology, particularly with respect to MCMC simulations, please refer to [Loomis _et al._ 2020](https://arxiv.org/abs/2009.11900)

If you use `molsim` for your analysis, please cite the Zenodo entry: [![DOI](https://zenodo.org/badge/253506425.svg)](https://zenodo.org/badge/latestdoi/253506425)

## Setup instructions

For science-ready code, we recommend downloading one of the releases—these are verified to the best of our ability to be accurate. If there are any indications otherwise, please submit an issue!

For the latest build, either for testing or for contributing, please clone the `development` branch of this reposity.

We recommend using `conda` for maintaining Python environments. Once you have acquired the code either from downloading a release or cloning the repository, you can create a new `conda` environment (called `molsim`) by running the following in the `molsim` directory:

`conda env create -f conda.yml molsim`

followed by:

`pip install .`

which will then install `molsim` into your Anaconda/Python installation. For developers/testers, we recommend you install using `pip install -e .`, which will create a symlink to the package to allow you to update code without having to constantly upgrade with `pip`.

## Contributions

The large majority of `molsim` was written by @bmcguir2, building on top of earlier code (`simulate_lte`).

The `mcmc` module was written by @laserkelvin, based heavily on earlier code by Dr. Ryan Loomis (see [his repo here](https://github.com/ryanaloomis/TMC1_mcmc_fitting)), which was used for the DR1 GOTHAM data analysis.

Any issues, please submit an issue, reporting what you think should happen and what actually happens.
