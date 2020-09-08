
from molsim.mcmc.base import GaussianLikelihood, UniformLikelihood, SingleComponent, MultiComponent

import numpy as np


def test_likelihoods():
    source_size = UniformLikelihood.from_values("ss", 0., 400.)
    ncol = UniformLikelihood.from_values("ncol", 0., 1e16)
    vlsr = UniformLikelihood.from_values("vlsr", 0., 10.)
    Tex = GaussianLikelihood.from_values("tex", 5.8, 0.5, 0., 10.)
    dV = GaussianLikelihood.from_values("dV", 0.1, 1e-1, 0., 0.3)

    uniform_test = source_size.ln_likelihood(100.)
    assert uniform_test == 0.

    normal_test = Tex.ln_likelihood(5.87)
    assert np.round(
        np.abs(-0.23559135 - normal_test), 4
    ) == 0.
    
    fail_test = dV.ln_likelihood(-5.)
    assert not np.isfinite(fail_test)


def test_single_component():
    source_size = UniformLikelihood.from_values("ss", 0., 400.)
    ncol = UniformLikelihood.from_values("ncol", 0., 1e16)
    vlsr = UniformLikelihood.from_values("vlsr", 0., 10.)
    Tex = GaussianLikelihood.from_values("tex", 5.8, 0.5, 0., 10.)
    dV = GaussianLikelihood.from_values("dV", 0.1, 1e-1, 0., 0.3)

    model = SingleComponent(
        source_size,
        vlsr,
        ncol,
        Tex,
        dV,
        None,
        None,
        100.,
        2.725
        )


def test_multi_component():
    source_sizes = [UniformLikelihood.from_values("ss", 0., 400.) for _ in range(4)]
    vlsrs = [UniformLikelihood.from_values("vlsr", 0., 10.) for _ in range(4)]
    ncols = [UniformLikelihood.from_values("ncol", 0., 1e16) for _ in range(4)]
    Tex = GaussianLikelihood.from_values("tex", 5.8, 0.5, 0., 10.)
    dV = GaussianLikelihood.from_values("dV", 0.1, 1e-1, 0., 0.3)

    model = MultiComponent(
        source_sizes,
        vlsrs,
        ncols,
        Tex,
        dV,
        None,
        None,
        100.,
        2.725
        )
