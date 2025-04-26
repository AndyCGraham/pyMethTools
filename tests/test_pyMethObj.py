import pytest
import numpy as np
import pandas as pd
import os
import sys
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, scripts_path)
from pyMethTools.pyMethObj import pyMethObj

@pytest.fixture
def sample_data():
    np.random.seed(1)
    meth = np.array(
        [[17., 15., 16., 18.,  8., 16., 19., 14., 15., 13., 10., 18., 16.,
            13., 16., 15., 13., 15., 15., 13., 20., 16., 13., 11., 14., 15.,
            18., 15., 14., 17.],
        [ 2.,  1.,  1.,  2.,  2.,  3.,  1.,  2.,  2.,  0.,  2.,  1.,  0.,
            1.,  1.,  4.,  5.,  1.,  4.,  4.,  4.,  3.,  8.,  6.,  6.,  7.,
            5.,  3.,  6.,  6.],
        [ 0.,  1.,  0.,  3.,  2.,  1.,  1.,  3.,  1.,  1.,  2.,  2.,  0.,
            2.,  0.,  7.,  5.,  3.,  4.,  4.,  6.,  5.,  4., 10.,  5.,  4.,
            3.,  3.,  4.,  3.],
        [ 0.,  2.,  1.,  0.,  2.,  0.,  2.,  1.,  4.,  5.,  0.,  2.,  3.,
            2.,  2.,  4.,  4.,  3.,  8.,  3.,  6.,  5., 10.,  3.,  5., 10.,
            8.,  7.,  4.,  4.],
        [11., 12., 10., 12., 17., 17., 13., 11., 19., 13., 15., 15., 16.,
            18., 13., 16., 17., 14., 14., 12., 16., 11., 13., 13., 13., 12.,
            11., 14., 12., 14.],
        [25., 27., 24., 29., 28., 26., 29., 26., 27., 26., 27., 26., 30.,
            26., 30., 26., 29., 25., 27., 25., 25., 23., 29., 26., 28., 27.,
            27., 27., 22., 25.],
        [ 3.,  2.,  2.,  2.,  2.,  3.,  2.,  2.,  3.,  4.,  3.,  2.,  1.,
            3.,  3.,  2.,  2.,  3.,  1.,  1.,  2.,  1.,  2.,  1.,  0.,  0.,
            6.,  2.,  2.,  3.],
        [ 4.,  3.,  3.,  2.,  1.,  4.,  1.,  4.,  2.,  2.,  1.,  1.,  4.,
            1.,  1.,  1.,  5.,  3.,  0.,  2.,  4.,  4.,  2.,  2.,  1.,  6.,
            1.,  5.,  2.,  2.],
        [ 0.,  4.,  2.,  2.,  5.,  3.,  6.,  5.,  2.,  4.,  4.,  5.,  0.,
            2.,  6.,  3.,  2.,  1.,  2.,  1.,  4.,  2.,  1.,  6.,  5.,  5.,
            7.,  1.,  2.,  4.],
        [20., 25., 23., 21., 23., 24., 22., 27., 27., 25., 24., 20., 26.,
            25., 20., 22., 26., 21., 25., 25., 23., 23., 24., 26., 21., 24.,
            26., 22., 18., 20.],
        [15., 17., 17., 18., 19., 19., 17., 17., 17., 17., 13., 17., 18.,
            20., 14., 18., 16., 17., 17., 18., 17., 19., 20., 16., 16., 16.,
            16., 16., 13., 17.],
        [ 2.,  4.,  2.,  2.,  1.,  1.,  6.,  3.,  3.,  4.,  2.,  3.,  4.,
            7.,  3.,  7.,  3.,  6.,  6.,  6.,  7.,  7.,  8.,  9.,  9.,  9.,
            11.,  2.,  5.,  2.],
        [ 2.,  1.,  7.,  4.,  3.,  2.,  5.,  2.,  1.,  4.,  3.,  3.,  5.,
            3.,  1.,  8., 10.,  5., 11.,  4., 10.,  6.,  4.,  7.,  6.,  6.,
            6.,  9.,  7.,  6.],
        [ 2.,  4.,  5.,  2.,  3.,  1.,  2.,  1.,  2.,  1.,  1.,  4.,  2.,
            6.,  6.,  7.,  8.,  5.,  6.,  4.,  5.,  8.,  5.,  7.,  8.,  4.,
            7.,  5., 11.,  7.],
        [14., 12., 19., 17., 16., 16., 18., 13., 18., 18., 18., 18., 17.,
            16., 15., 14., 16., 20., 15., 11., 15., 14., 12., 18., 18., 15.,
            16., 12., 12., 18.]]
    )
    coverage = np.array(
        [[30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 30., 28., 28., 29., 29., 29., 28., 29., 29., 28., 29.,
            29., 28., 29., 29.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 30., 28., 28., 29., 29., 29., 28., 29., 29., 28., 29.,
            29., 28., 29., 29.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 30., 28., 28., 29., 29., 29., 28., 29., 29., 28., 29.,
            29., 28., 29., 29.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 30., 28., 28., 29., 29., 29., 28., 29., 29., 28., 29.,
            29., 28., 29., 29.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 30., 28., 28., 29., 29., 29., 28., 29., 29., 28., 29.,
            29., 28., 29., 29.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 29., 28., 29., 29., 28., 29., 30., 29., 29., 28.,
            29., 30., 27., 28.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 29., 28., 29., 29., 28., 29., 30., 29., 29., 28.,
            29., 30., 27., 28.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 29., 28., 29., 29., 28., 29., 30., 29., 29., 28.,
            29., 30., 27., 28.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 29., 28., 29., 29., 28., 29., 30., 29., 29., 28.,
            29., 30., 27., 28.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 29., 28., 29., 29., 28., 29., 30., 29., 29., 28.,
            29., 30., 27., 28.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 30., 28., 29., 28., 28., 29., 28., 28., 29., 29.,
            28., 27., 28., 30.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 30., 28., 29., 28., 28., 29., 28., 28., 29., 29.,
            28., 27., 28., 30.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 30., 28., 29., 28., 28., 29., 28., 28., 29., 29.,
            28., 27., 28., 30.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 30., 28., 29., 28., 28., 29., 28., 28., 29., 29.,
            28., 27., 28., 30.],
        [30., 29., 29., 29., 29., 30., 30., 28., 29., 30., 29., 29., 30.,
            30., 30., 29., 30., 28., 29., 28., 28., 29., 28., 28., 29., 29.,
            28., 27., 28., 30.]]
            )
    target_regions = np.repeat([1,2,3], 5)
    genomic_positions = np.array(range(200,6600,450))
    covs = pd.DataFrame({'intercept': np.ones(30), 'cov1': np.repeat([0,1],15)})
    covs_disp = pd.DataFrame({'intercept': np.ones(30)})
    return meth, coverage, target_regions, genomic_positions, covs, covs_disp

@pytest.fixture
def fitted_obj(sample_data):
    np.random.seed(1)
    meth, coverage, target_regions, genomic_positions, covs, covs_disp = sample_data
    obj = pyMethObj(meth, coverage, target_regions, genomic_positions, covs=covs, covs_disp=covs_disp)
    # Predefine fits for testing
    obj.fits = [
    np.array([[ 7.92886322e-01,  1.25959074e-02,  1.00000000e-15],
            [ 2.01046223e-01,  2.12963479e-01,  1.00000000e-15],
            [ 1.74930051e-01,  2.34600036e-01,  1.00000000e-15],
            [ 2.05591731e-01,  2.43891601e-01,  1.00000000e-15],
            [ 7.65952286e-01, -1.20519681e-02,  1.00000000e-15]]),
    np.array([[ 1.32184021e+00, -4.11999233e-02,  1.00000000e-15],
            [ 2.90850673e-01, -5.85198271e-02,  1.00000000e-15],
            [ 2.70992165e-01,  1.72875207e-02,  1.00000000e-15],
            [ 3.14645981e-01,  2.61748915e-03,  1.00000000e-15],
            [ 1.11590907e+00,  1.27013657e-03,  1.00000000e-15]]),
    np.array([[ 8.64444799e-01,  1.08287909e-02,  1.00000000e-15],
            [ 3.22801695e-01,  1.63522972e-01,  1.00000000e-15],
            [ 3.17690976e-01,  1.96341136e-01,  1.00000000e-15],
            [ 3.00396400e-01,  1.92360772e-01,  1.00000000e-15],
            [ 8.41474308e-01, -2.73828372e-02,  1.00000000e-15]])             
            ]
    obj.se = [
    np.array([[0.04761905, 0.067733  ],
        [0.04761905, 0.067733  ],
        [0.04761905, 0.067733  ],
        [0.04761905, 0.067733  ],
        [0.04761905, 0.067733  ]]),
    np.array([[0.04761905, 0.067733  ],
            [0.04761905, 0.067733  ],
            [0.04761905, 0.067733  ],
            [0.04761905, 0.067733  ],
            [0.04761905, 0.067733  ]]),
    np.array([[0.04761905, 0.06785295],
            [0.04761905, 0.06785295],
            [0.04761905, 0.06785295],
            [0.04761905, 0.06785295],
            [0.04761905, 0.06785295]])
    ]
    obj.link = "arcsin"
    obj.fit_method = "gls"
    return obj

def test_init(sample_data):
    meth, coverage, target_regions, genomic_positions, covs, covs_disp = sample_data
    obj = pyMethObj(meth, coverage, target_regions, genomic_positions, covs=covs, covs_disp=covs_disp)
    assert obj.meth.shape == meth.shape
    assert obj.coverage.shape == coverage.shape
    assert obj.ncpgs == meth.shape[0]
    assert np.array_equal(obj.target_regions, target_regions)
    assert np.array_equal(obj.genomic_positions, genomic_positions)
    assert np.array_equal(obj.X, covs.to_numpy())
    assert np.array_equal(obj.X_star, covs_disp.to_numpy())

def test_fit_betabinom(sample_data):
    meth, coverage, target_regions, genomic_positions, covs, covs_disp = sample_data
    obj = pyMethObj(meth, coverage, target_regions, genomic_positions, covs=covs, covs_disp=covs_disp)
    obj.fit_betabinom(ncpu=1)
    assert len(obj.fits) == len(obj.individual_regions)

def test_fit_cpg_local(fitted_obj):
    result,se = fitted_obj.fit_cpg_local(0)
    assert result is not None
    assert len(result) == fitted_obj.X.shape[1] + fitted_obj.X_star.shape[1]

def test_fit_region_local(fitted_obj):
    cpgs=np.array(range(fitted_obj.ncpgs))
    result = fitted_obj.fit_region_local(cpgs, 1)
    assert result is not None
    assert len(result[0]) == 5  # fits and se
    assert len(result[0][0]) == fitted_obj.X.shape[1] + fitted_obj.X_star.shape[1]

def test_smooth(fitted_obj):
    fits=fitted_obj.fits
    fitted_obj.smooth(ncpu=1)
    assert len(fitted_obj.fits) == len(fits)

def test_wald_test(fitted_obj):
    cpg_res,region_res = fitted_obj.wald_test('cov1')
    assert 'pvals' in cpg_res.columns
    assert 'fdrs' in cpg_res.columns
    assert len(cpg_res) == fitted_obj.ncpgs
    assert all(cpg_res.loc[cpg_res.fdrs < 0.05,"pos"] == [650,1100,1550,5150,5600,6050])
    assert 'chr' in region_res.columns
    assert 'start' in region_res.columns
    assert 'end' in region_res.columns
    assert 'num_cpgs' in region_res.columns
    assert 'num_sig_cpgs' in region_res.columns
    assert len(region_res) == 2
    assert all(region_res["start"] == [650,5150])
    assert all(region_res["end"] == [1550,6050])

def test_permute_and_refit(fitted_obj):
    result = fitted_obj.permute_and_refit('cov1', N=10)
    assert 'stats' in result
    assert 'pvals' in result
    assert 'fdrs' in result
    assert result['stats'].shape[1] == 10  # N permutations

def test_find_codistrib_regions(fitted_obj):
    np.random.seed(42)
    fitted_obj.find_codistrib_regions()
    print(fitted_obj.codistrib_regions)
    assert all(np.array(['0', '1_1-3', '1_1-3', '1_1-3', '4', '0', '2_1-3', '2_1-3',
       '2_1-3', '4', '0', '3_1-3', '3_1-3', '3_1-3', '4'], dtype='<U32') == fitted_obj.codistrib_regions)

def test_sim_multiple_cpgs(fitted_obj):
    fitted_obj.codistrib_regions = np.array(['0', '1_1-3', '1_1-3', '1_1-3', '4', '0', '2_1-3', '2_1-3',
       '2_1-3', '4', '0', '3_1-3', '3_1-3', '3_1-3', '4'])
    sim_meth,sim_coverage,adjust,adjust_regions = fitted_obj.sim_multiple_cpgs(
                                                        sample_size=15,
                                                        use_codistrib_regions=True,
                                                        ncpu=1,adjust_factor=0.15,
                                                        n_diff_regions=[2,0],
                                                        chunksize=1)
    np.random.seed(42)
    assert sim_meth.shape == sim_coverage.shape
    assert np.allclose(sim_meth[adjust==0].mean(), fitted_obj.meth[adjust==0].mean(), atol=10)
    assert np.allclose(sim_coverage.mean(), fitted_obj.coverage.mean(), atol=10)
    assert sim_meth.shape[0] == fitted_obj.meth.shape[0]  # Same ncpgs as data
    assert sim_meth.shape[1] == 15  # default sample size
    assert all(sim_meth[adjust==0.15].mean(axis=1) > fitted_obj.meth[adjust==0.15].mean(axis=1))

def test_region_plot(fitted_obj):
    fitted_obj.codistrib_regions = np.array(['0', '1_1-3', '1_1-3', '1_1-3', '4', '0', '2_1-3', '2_1-3',
       '2_1-3', '4', '0', '3_1-3', '3_1-3', '3_1-3', '4'])
    fitted_obj.region_plot(1)
    assert True

def test_copy(fitted_obj):
    obj_copy = fitted_obj.copy()
    assert obj_copy is not fitted_obj
    assert obj_copy.meth.shape == fitted_obj.meth.shape
    assert np.array_equal(obj_copy.meth, fitted_obj.meth)
    assert np.array_equal(obj_copy.coverage, fitted_obj.coverage)
    assert np.array_equal(obj_copy.target_regions, fitted_obj.target_regions)
    assert np.array_equal(obj_copy.genomic_positions, fitted_obj.genomic_positions)
    assert np.array_equal(obj_copy.X, fitted_obj.X)
    assert np.array_equal(obj_copy.X_star, fitted_obj.X_star)
    assert np.array_equal(obj_copy.fits, fitted_obj.fits)