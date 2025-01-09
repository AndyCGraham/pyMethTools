import pytest
from pyMethTools.fit import *
import numpy as np
import scipy

np.random.seed(30)

@pytest.fixture
def meth_levels():
    return [0.5,0.06,0.03,0.06,0.5,0.9,0.1,0.1,0.1,0.8, 0.6,0.1,0.1,0.1,0.5]

@pytest.fixture
def fits(meth_levels):
    sample_size=15
    n_cpg=15
    coverage = np.empty(shape=(n_cpg,sample_size), dtype=int)
    meth = np.empty(shape=(n_cpg,sample_size), dtype=int)
    for sample in range(0, sample_size):
        n=int(np.random.normal(30,5))
        coverage[:,sample]=np.repeat(n,n_cpg)
        for cpg,k in zip(range(0, n_cpg+1),meth_levels):
            meth[cpg,sample]=np.random.binomial(n, k, 1)[0]
    regions = np.repeat([1,2,3], 5)
    fits = fit_betabinom(meth, coverage, regions)
    return fits

def test_betabinom(fits):
    """
    Test that the coorect class is returned
    """
    assert isinstance(fits, np.ndarray) 

def test_betabinom_contents(fits):
    """
    Test that the correct internal classes are returned
    """
    assert isinstance(fits[0], scipy.optimize._optimize.OptimizeResult) 

def test_fits(fits,meth_levels):
    """
    Test that inferred mu is fairly accurate
    """
    assert all(abs(np.array([fit.x[0]/(fit.x[0]+fit.x[1]) for fit in fits]) - np.array(meth_levels)) < 0.1)