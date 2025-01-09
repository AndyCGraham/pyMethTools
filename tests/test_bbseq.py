import pytest
from pyMethTools.bbseq import *
import numpy as np
import scipy
import pandas as pd
from numpy import genfromtxt

np.random.seed(100)

@pytest.fixture
def coverage():
    return genfromtxt("tests/test_data/template_coverage_diff.csv", delimiter=',')

@pytest.fixture
def meth():
    return genfromtxt("tests/test_data/template_meth_diff.csv", delimiter=',')

@pytest.fixture
def covs():
    X=np.vstack([np.repeat(1,30),np.repeat([0,1],15)]).T
    X=pd.DataFrame(X)
    X.columns=['intercept','group']
    X_star = X.drop(columns='group')
    return X,X_star

@pytest.fixture
def bbseq_cpg(coverage,meth,covs):
    return bbseq(meth,coverage,cometh=[],covs=covs[0],covs_disp=covs[1],ncpu=1,dmrs=False)

@pytest.fixture
def bbseq_dmr(coverage,meth,covs):
    return bbseq(meth,coverage,target_regions=np.repeat([1,2,3], 5),cometh=[],covs=covs[0],covs_disp=covs[1],ncpu=1)

@pytest.fixture
def bbseq_cometh(coverage,meth,covs):
    return bbseq(meth,coverage,cometh=np.array([1.,1.1,1.1,1.1,1.,2.,2.1,2.1,2.1,2.,3.,3.1,3.1,3.1,3.],dtype='float32'),
                 covs=covs[0],covs_disp=covs[1],ncpu=1,dmrs=False)
    
def test_bbseq_cpg_class(bbseq_cpg):
    assert isinstance(bbseq_cpg, pd.core.frame.DataFrame) 
    
def test_bbseq_dmr_class(bbseq_dmr):
    assert isinstance(bbseq_dmr[0], pd.core.frame.DataFrame) 

def test_bbseq_cometh_class(bbseq_cometh):
    assert isinstance(bbseq_cometh, pd.core.frame.DataFrame) 

def test_bbseq_cpg_size(bbseq_cpg,meth,covs):
    assert bbseq_cpg.shape[0] == meth.shape[0] * covs[0].shape[1]

def test_bbseq_cpg_dtype(bbseq_cpg):
    assert all(bbseq_cpg.dtypes == ['float64','float64','float64','float64','object'])

def test_bbseq_cpg_nan(bbseq_cpg):
    assert bbseq_cpg.isnull().values.any() == False

def test_dmr_cpg_res(bbseq_cpg,bbseq_dmr):
    assert bbseq_dmr[0].iloc[:,0:4].equals(bbseq_cpg.iloc[:,0:4])

def test_bbseq_dmr_dmrs(bbseq_dmr):
    assert bbseq_dmr[1].shape[0] == 4