import pytest
from pyMethTools.fit import *
from pyMethTools.comethyl import *
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
def coverage():
    return genfromtxt("tests/test_data/template_coverage.csv", delimiter=',')

@pytest.fixture
def meth():
    return genfromtxt("tests/test_data/template_meth.csv", delimiter=',')

@pytest.fixture
def cometh(fits,coverage,meth):
    cometh = find_comethyl_regions(fits,meth,coverage,target_regions=np.repeat([1,2,3], 5),min_cpgs=3)
    return cometh

def test_cometh_class(cometh):
    """
    Test that the coorect class is returned
    """
    assert isinstance(cometh, np.ndarray) 

def test_cometh_contents(cometh):
    """
    Test that the correct regions are found
    """
    assert all(cometh == np.array([1. , 1.1, 1.1, 1.1, 1. , 2. , 2.1, 2.1, 2.1, 2. , 3. , 3.1, 3.1,
       3.1, 3. ], dtype='float32'))