import pytest
import pyMethTools
from pyMethTools.pyMethObj import *
import numpy as np
from numpy import genfromtxt
import scipy
from scipy.special import expit

np.random.seed(1)

@pytest.fixture
def meth_levels():
    return[0.5,0.06,0.06,0.06,0.5,0.9,0.1,0.1,0.1,0.8, 0.6,0.12,0.13,0.14,0.5]

@pytest.fixture
def data(meth_levels):
    sample_size=15
    n_cpg=15
    coverage = np.empty(shape=(n_cpg,sample_size), dtype=int)
    meth = np.empty(shape=(n_cpg,sample_size), dtype=int)
    for sample in range(0, sample_size):
        n=int(np.random.normal(30,1))
        coverage[:,sample]=np.repeat(n,n_cpg)
        for cpg,k in zip(range(0, n_cpg+1),meth_levels):
            meth[cpg,sample]=np.random.binomial(n, k, 1)[0]
    regions = np.repeat([1,2,3], 5)
    return meth,coverage,regions

@pytest.fixture
def obj(data):
    meth,coverage,regions=data
    obj = pyMethObj(meth,coverage,regions)
    return obj

@pytest.fixture
def fit(obj):
    obj.fit_betabinom()
    return obj

@pytest.fixture
def bbseq(fit):
    fit.bbseq()
    return fit

@pytest.fixture
def cometh(bbseq):
    cometh = bbseq.codistrib_regions
    return cometh

@pytest.fixture
def sim(bbseq):
    return bbseq.sim_multiple_cpgs(sample_size=15,use_codistrib_regions=True,ncpu=1,adjust_factor=[0.05,0.025],n_diff_regions=2)

@pytest.fixture
def diff_data():
    meth=genfromtxt("tests/test_data/template_meth.csv", delimiter=',')
    coverage=genfromtxt("tests/test_data/template_coverage.csv", delimiter=',')
    regions=genfromtxt("tests/test_data/template_regions.csv", delimiter=',')
    regions=regions.astype(int)
    X = pd.read_csv("tests/test_data/template_X.csv",index_col=0)
    X_star = pd.read_csv("tests/test_data/template_X_star.csv",index_col=0)
    return meth,coverage,regions,X,X_star

@pytest.fixture
def bbseq_res(diff_data):
    obj=pyMethObj(diff_data[0],diff_data[1],diff_data[2],diff_data[3],diff_data[4])
    obj.fit_betabinom()
    obj.bbseq()
    return obj

def test_obj(obj):
    """
    Test that the coorect class is returned
    """
    assert isinstance(obj, pyMethTools.pyMethObj.pyMethObj) 

def test_obj_contents(fit,data):
    """
    Test that internal data is integrated correctly
    """
    meth,coverage,regions=data
    assert np.equal(fit.meth,meth).all()

def test_fits(fit,meth_levels):
    """
    Test that inferred mu is fairly accurate
    """
    assert all(([expit(fit.theta[0]) for fit in fit.fits] - np.array(meth_levels)) < 0.1)

def test_cometh_class(cometh):
    """
    Test that the coorect class is returned
    """
    assert isinstance(cometh, np.ndarray) 

def test_cometh_contents(cometh):
    """
    Test that the correct regions are found
    """
    assert all(cometh == np.array(['0', '1_1-3', '1_1-3', '1_1-3', '4', '5', '2_1-3', '2_1-3',
       '2_1-3', '9', '10', '3_1-3', '3_1-3', '3_1-3', '14'], dtype='str'))

def test_sim_meth_class(sim):
    assert isinstance(sim[0], np.ndarray) 

def test_sim_coverage_class(sim):
    assert isinstance(sim[1], np.ndarray) 

def test_adjust_class(sim):
    assert isinstance(sim[2], np.ndarray) 

def test_adjust_unchanged(sim):
    assert sum(sim[2] == 0) == 9

def test_adjust_pos(sim):
    assert sum(sim[2] == 0.05) == 3

def test_adjust_neg(sim):
    assert sum(sim[2] == -0.025) == 3

def test_meth_unchanged(sim,data):
    assert all(abs(sim[0][sim[2]==0,:].mean(axis=1) - data[0][sim[2]==0,:].mean(axis=1))) < 5

def test_meth_increased(sim,data):
    assert all(sim[0][sim[2]==0.05,:].mean(axis=1) - data[0][sim[2]==0.05,:].mean(axis=1) > 0)

def test_meth_reduced(sim,data):
    assert all(sim[0][sim[2]==-0.025,:].mean(axis=1) - data[0][sim[2]==-0.025,:].mean(axis=1) < 0)

def test_bbseq_cpg(bbseq_res):
    """
    Test that differentially methylated cpgs are picked up
    """
    cpg_res = bbseq_res.get_contrast(contrast="group")
    assert all(cpg_res["sig"] == [False,False,False,False,False,False,True,True,True,False,False,True,True,True,False])

def test_bbseq_region(bbseq_res):
    """
    Test that differentially methylated cpgs are picked up
    """
    region_res = bbseq_res.get_contrast("region",contrast="group")
    assert all(region_res["sig"] == [False,True,True])

def test_bbseq_region_codistrib(bbseq_res):
    """
    Test that differentially methylated cpgs are picked up
    """
    region_res = bbseq_res.get_contrast("region",contrast="group")
    assert all(region_res["site"] == ['1_1-3','2_1-3','3_1-3'])
