
from molsim.mcmc.base import GaussianLikelihood, UniformLikelihood
from molsim.mcmc.models import SingleComponent, MultiComponent

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
    seed = np.random.seed(42)

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
        None,   # no molecule required
        None    # no observation required
        )
    # check the intialization routine is still working
    initial = model.sample_prior()
    static = [1.49816048e+02, 9.50714306e+00, 7.31993942e+15, 5.24405994e+00, 1.31890218e-01]
    assert np.allclose(initial, static)


def test_multi_component():
    seed = np.random.seed(42)

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
        )

    # should be 14 parameters total; 3 * 4 components + 2
    assert len(model) == 14

    model._get_components()

    # check the intialization routine is still working deterministically when asked to be
    initial = model.sample_prior()
    static = [149.81604753894499, 62.39780813448106, 8.233797718320979, 73.36180394137352, 9.50714306409916, 0.5808361216819946, 9.699098521619943, 3.0424224295953772, 7319939418114051.0, 8661761457749352.0, 8324426408004217.0, 5247564316322378.0, 4.937541083743484, 0.043771247075902735]
    assert np.allclose(initial, static)

    # make sure the combined likelihood looks reasonable
    parameters = [100., 96, 20, 45, 5., 5.6, 6.44, 4.3, 1e10, 1e11, 1e9, 1e11, 5.87, 0.09216]
    likelihood = model.compute_prior_likelihood(parameters)
    assert np.round(abs(likelihood - 4.579927708578582)) == 0.
