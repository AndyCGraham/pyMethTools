import pytest
from pyMethTools.sim import *
import numpy as np
import scipy
from numpy import genfromtxt
import pickle

np.random.seed(30)

@pytest.fixture
def fits():
    with open("tests/test_data/template_fits.csv", 'rb') as f:
        fits = pickle.load(f)
    return fits

@pytest.fixture
def meth():
    return genfromtxt("tests/test_data/template_meth.csv", delimiter=',')

@pytest.fixture
def sim(fits):
    return sim_multiple_cpgs(fits,adjust_factor=2,cometh_regions=np.array([1.,1.1,1.1,1.1,1.,2.,2.1,2.1,2.1,2.,3.,3.1,3.1,3.1,3.]),
                             n_diff_regions=2,vary_read_depth=False,sample_size=15)

def test_sim_meth_class(sim):
    assert isinstance(sim[0], np.ndarray) 

def test_sim_coverage_class(sim):
    assert isinstance(sim[1], np.ndarray) 

def test_adjust_class(sim):
    assert isinstance(sim[2], np.ndarray) 

def test_adjust_unchanged(sim):
    assert sum(sim[2] == 0) == 9

def test_adjust_pos(sim):
    assert sum(sim[2] == 2) == 3

def test_adjust_neg(sim):
    assert sum(sim[2] == -2) == 3

def test_meth_unchanged(sim,meth):
    assert all(abs(sim[0][sim[2]==0,:].mean(axis=1) - meth[sim[2]==0,:].mean(axis=1))) < 5

def test_meth_increased(sim,meth):
    assert all(sim[0][sim[2]==2,:].mean(axis=1) - meth[sim[2]==2,:].mean(axis=1) > 0)

def test_meth_reduced(sim,meth):
    assert all(sim[0][sim[2]==-2,:].mean(axis=1) - meth[sim[2]==-2,:].mean(axis=1) < 0)