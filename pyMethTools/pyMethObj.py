import numpy as np
from numba import njit
import pandas as pd
from pandas.api.types import is_integer_dtype
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta,logit,expit,digamma,polygamma
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.families.links import Link
from itertools import chain
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
from pyMethTools.fit_cpg import *
import numpy as np
import pandas as pd
from hmmlearn import hmm
from pandas.api.types import is_integer_dtype
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta,logit,expit,digamma,polygamma
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.families.links import Link
from itertools import chain
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy

class pyMethObj():
    """
    A Python class containing functions for analysis of targetted DNA methylation data.
    These include:
        Fitting beta binomial models to cpgs, inferring parameters for the mean abundance, dispersion, and any other given covariates: fit_betabinom().
        Smoothing fitted parameters: smooth().
        Differential methylation analysis by assessing the statistical significance of covariate parameters, while finding contigous regions of codistributed 
        cpgs with similar parameters, and testing if these are differentially methylated: bbseq().
        Getting regression results for any contrast: get_contrast().
        Simulating new data based on fitted data: sim_multiple_cpgs().

    All methods allow parallel processing to speed up computation (set ncpu parameter > 1).
    
    Beta binomial fitting to CpGs is based on a method implemented by pycorncob (https://github.com/jgolob/pycorncob/tree/main), and concieved by 
    https://doi.org/10.1214/19-AOAS1283, with a few adjustments for DNA methylation data.
    """
    
    def __init__(self,meth: np.ndarray,coverage: np.ndarray,target_regions: np.ndarray,
                 genomic_positions=None,chr=None,covs=None,covs_disp=None,
                 phi_init: float=0.5,maxiter: int=500,maxfev: int=500,sample_weights=None):
        """
        Intitiate pyMethObj Class

        Parameters:
            meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
            target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
            genomic_positions (1D Numpy array or None, default: None): Array of same length as number of cpgs in meth/coverage, specifying the genomic position of each cpg on the chromosome.
            covs (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the mean parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            covs_disp (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column
            phi_init (float, default: 0.5): Initial value of the dispersion parameter of the beta binomial model fit to each cpg.
            maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
            maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        """

        assert meth.shape == coverage.shape, "'meth' and 'coverage' must have the same shape"
        assert meth.shape[0] == len(target_regions), "Length of 'target_regions' should be equal to the number of rows (cpgs) in meth"
        if genomic_positions is not None:
            assert meth.shape[0] == len(genomic_positions), "Length of 'genomic_positions' should be equal to the number of rows (cpgs) in meth"
            isinstance(genomic_positions,np.ndarray), "'genomic_positions' should be a numpy array of integer positions"
            assert genomic_positions.dtype == int, "'genomic_positions' should be a numpy array of integer positions"
        if covs is not None:
            assert meth.shape[1] == covs.shape[0], "Covs should have one row for each sample (column) in meth/coverage"
            assert any([isinstance(covs,np.ndarray),pd.DataFrame]), "'covs' should be a numpy array or pandas dataframe"
        assert isinstance(maxiter,int), "maxiter must be positive integer"
        assert maxiter>0, "maxiter must be positive integer"
        assert isinstance(maxfev,int), "maxfev must be positive integer"
        assert maxfev>0, "maxfev must be positive integer"
        
        self.meth = meth.astype('float64')
        self.coverage = coverage.astype('float64')
        self.ncpgs = meth.shape[0]
        self.target_regions = target_regions
        self.individual_regions, self.region_cpg_counts = np.unique(target_regions, return_counts=True)
        self.new_region_indices = self.region_cpg_counts.cumsum()[:-1]
        self.individual_regions = np.array(self.unique(target_regions))
        counts = {}
        self.region_cpg_indices = np.array([counts[x]-1 for x in target_regions if not counts.update({x: counts.get(x, 0) + 1})])
        self.genomic_positions = genomic_positions
        if chr is not None:
            self.chr = chr
        else:
            self.chr = np.repeat("chr",len(genomic_positions))

        if not isinstance(covs, pd.DataFrame):
            covs = pd.DataFrame(np.repeat(1,meth.shape[1]))
            covs.columns=['intercept']

        assert covs.shape[0] == meth.shape[1], "covs must have one row per sample (column in meth)"
            
        if not isinstance(covs_disp, pd.DataFrame):
            covs_disp = pd.DataFrame(np.repeat(1,meth.shape[1]))
            covs_disp.columns=['intercept']

        assert covs_disp.shape[0] == meth.shape[1], "covs_disp must have one row per sample (column in meth)"
        
        self.X = covs.to_numpy().astype('float64')
        self.X_star = covs_disp.to_numpy().astype('float64')
        self.param_abd = covs.columns
        self.n_param_abd = len(covs.columns)
        self.param_disp = covs_disp.columns
        self.n_param_disp = len(covs_disp.columns)
        self.n_ppar = len(covs.columns) + len(covs_disp.columns)
        self.df_model = self.n_ppar
        self.df_residual = len(covs) - self.df_model
        
        if (self.df_residual) < 0:
            raise ValueError("Model overspecified. Trying to fit more parameters than sample size.")
        
        self.param_names_abd = covs.columns
        self.param_names_disp = covs_disp.columns
        self.param_names = np.hstack([self.param_names_abd,self.param_names_disp])
        self.phi_init = phi_init
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.sample_weights = sample_weights
        
        # Inits
        self.fits=[]
        self.codistrib_regions=[]
        self.codistrib_region_min_beta=[]
        self.codistrib_region_max_beta=[]
        self.beta_vals = []
        self.region_cov = []
        self.region_cov_sd = []
        self.disp_intercept = []

    def fit_betabinom(self,ncpu: int=1,start_params=[],maxiter=None,maxfev=None,link="arcsin",fit_method="gls",chunksize=1):
        """
        Fit beta binomial model to DNA methylation data.
    
        Parameters:
            chunksize (integer, default: 1): Number of regions to process at once if using parallel processing.
            ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs. 
        """

        assert isinstance(ncpu,int), "ncpu must be positive integer"
        assert ncpu>0, "ncpu must be positive integer"

        #Use object defaults for maxiter/maxfev if not given
        if maxiter is None:
            maxiter=self.maxiter
        if maxfev is None:
            maxfev=self.maxfev
        
        assert maxiter>0, "maxiter must be positive integer"
        assert isinstance(maxfev,int), "maxfev must be positive integer"
        assert maxfev>0, "maxfev must be positive integer"
        assert any([link=="arcsin",link=="logit"]), "link must be 'arcsin' or 'logit'"

        cpgs=np.array(range(self.ncpgs))
        self.link = link
        self.fit_method=fit_method
    
        if ncpu > 1: #Use ray for parallel processing
            ray.init(num_cpus=ncpu)    
            X_id=ray.put(self.X)
            X_star_id=ray.put(self.X_star)
            if chunksize>1:
                self.fits,self.se = zip(*ray.get([self.fit_betabinom_chunk.remote(
                    self.target_regions[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.meth[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.coverage[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    X_id,X_star_id,self.fit_region_internal,self.fit_cpg_internal,self.unique,
                    maxiter,maxfev,start_params,
                    self.param_names_abd,self.param_names_disp,self.link,
                    self.fit_method, self.sample_weights) 
                              for chunk in range(0,len(set(self.individual_regions))+1,chunksize)]))
                
                self.fits = list(chain.from_iterable(self.fits))
                self.se = list(chain.from_iterable(self.se))
             
            else:
                self.fits,self.se = zip(*ray.get([self.fit_region.remote(self.region_cpg_indices[self.target_regions==region],
                                                            self.meth[self.target_regions==region],self.coverage[self.target_regions==region],
                                                            X_id,X_star_id,self.fit_cpg_internal,maxiter,maxfev,start_params,self.param_names_abd,
                                                            self.param_names_disp,self.link,self.fit_method,self.sample_weights) 
                                  for region in self.individual_regions]))

            ray.shutdown()
            
        else:
            self.fits,self.se = zip(*[self.fit_region_local(cpgs,region,start_params) for region in self.individual_regions])


    @staticmethod
    @ray.remote
    def fit_betabinom_chunk(region_id,meth_id,coverage_id,X_id,X_star_id,fit_region,fit_cpg,unique,maxiter=150,maxfev=150,start_params=[],
                            param_names_abd=["intercept"],param_names_disp=["intercept"],link="arcsin",fit_method="gls",
                            sample_weights=None):
        """
        Fit beta binomial model to a chunk of target regions from DNA methylation data.
    
        Parameters:
            chunk_regions (numpy array): Array of region assignments for this chunk.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
            maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
            maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs in this chunk. 
        """
        fits,se = zip(*[fit_region(meth_id[region_id==region],coverage_id[region_id==region],
                           X_id,X_star_id,fit_cpg,maxiter,maxfev,start_params,param_names_abd,param_names_disp,link,fit_method,
                           sample_weights) 
                                  for region in unique(region_id)])
        return fits,se

    @staticmethod
    def fit_region_internal(meth_id,coverage_id,X_id,X_star_id,fit_cpg,maxiter=150,maxfev=150,start_params=[],param_names_abd=["intercept"],
                            param_names_disp=["intercept"],link="arcsin",fit_method="gls",sample_weights=None):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            cpg (integer, float, or string): Cpg row number in meth_id/coverage_id.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
            matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
            parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """
        fits,se = zip(*[fit_cpg(meth_id[cpg],coverage_id[cpg],X_id,X_star_id,maxiter,maxfev,start_params,param_names_abd,param_names_disp,
                                       link,fit_method,sample_weights) 
                              for cpg in range(meth_id.shape[0])])
        fits = np.vstack(fits)
        se = np.vstack(se)
        
        return fits,se

    @staticmethod
    @ray.remote
    def fit_region(cpgs,meth_id,coverage_id,X_id,X_star_id,fit_cpg,maxiter=150,maxfev=150,start_params=[],param_names_abd=["intercept"],
                   param_names_disp=["intercept"],link="arcsin",fit_method="gls",sample_weights=None):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            cpg (integer, float, or string): Cpg row number in meth_id/coverage_id.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
            matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
            parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """
        fits,se = zip(*[fit_cpg(meth_id[cpg],coverage_id[cpg],X_id,X_star_id,maxiter,maxfev,start_params,param_names_abd,param_names_disp,
                                       link,fit_method,sample_weights) 
                              for cpg in cpgs])
        
        return fits,se
    
    @staticmethod
    def fit_cpg_internal(meth,coverage,X,X_star,maxiter=150,maxfev=150,start_params=[],param_names_abd=["intercept"],param_names_disp=["intercept"],
                link="arcsin",fit_method="gls",sample_weights=None):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            cpg (integer, float, or string): Cpg row number in meth_id/coverage_id.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
            matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
            parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """
        cc = Corncob_2(
                    total=coverage,
                    count=meth,
                    X=X,
                    X_star=X_star,
                    param_names_abd=param_names_abd,
                    param_names_disp=param_names_disp,
                    link=link,
                    fit_method=fit_method,
                    sample_weights=sample_weights
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev,start_params=start_params)
        return e_m.x,e_m.se_beta0
        
    def fit_cpg_local(self,cpg,start_params=[]):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            region (integer, float, or string): Name of the region being analysed.
            site (integer, float, or string): Name of cpg.
            meth (2D numpy array): Count table of methylated reads at each cpg in the region (rows) for each sample (columns).
            coverage (2D numpy array): Count table of total reads at each cpg in the region (rows) for each sample (columns).
            X (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the mean parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            min_cpgs (integer, default: 3): Minimum length of a dmr.
            res (boolean, default: True): Whether to return regression results.
            LL (boolean, default: False): Whether to return model log likelihood.
            
            
        Returns:
            pandas dataframe (optional, depending in res parameter): Dataframe containing estimates of coeficents for the cpg. 
            float (optional, depending in LL parameter): Model log likelihood
        """
        cc = Corncob_2(
                    total=self.coverage[cpg][~np.isnan(self.meth[cpg])],
                    count=self.meth[cpg][~np.isnan(self.meth[cpg])],
                    X=self.X[~np.isnan(self.meth[cpg])],
                    X_star=self.X_star[~np.isnan(self.meth[cpg])],
                    param_names_abd=self.param_names_abd,
                    param_names_disp=self.param_names_disp,
                    link=self.link,
                    fit_method=self.fit_method,
                    sample_weights=self.sample_weights
                )
        e_m = cc.fit(maxiter=self.maxiter,maxfev=self.maxfev,start_params=start_params)
        return e_m.x,e_m.se_beta0

    def fit_region_local(self,cpgs,region,start_params=[]):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            cpg (integer, float, or string): Cpg row number in meth_id/coverage_id.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
            matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
            parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """

        fits,se = zip(*[self.fit_cpg_local(cpg,start_params) for cpg in cpgs[self.target_regions==region]])
        fits = np.vstack(fits)
        se = np.vstack(se)

        return fits,se

    @staticmethod
    @ray.remote
    def fit_cpg(cpg,meth,coverage,X,X_star,maxiter=150,maxfev=150,start_params=[],param_names_abd=["intercept"],param_names_disp=["intercept"],
                link="arcsin",fit_method="gls",sample_weights=None):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            cpg (integer, float, or string): Cpg row number in meth_id/coverage_id.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
            matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
            parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form only an intercept column.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """
        cc = Corncob_2(
                    total=coverage[cpg],
                    count=meth[cpg],
                    X=X,
                    X_star=X_star,
                    param_names_abd=param_names_abd,
                    param_names_disp=param_names_disp,
                    link=link,
                    fit_method=fit_method,
                    sample_weights=sample_weights
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev,start_params=start_params)
        return e_m.x,e_m.se_beta0

    def smooth(self,lambda_factor=10,param="intercept",ncpu: int=1,chunksize=1):

        if ncpu > 1:
            ray.init(num_cpus=ncpu)  
            if chunksize > 1:
                self.fits = ray.get([self.smooth_chunk.remote(
                    self.target_regions[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.fits[chunk:chunk+chunksize],self.n_param_abd,
                    self.genomic_positions[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.coverage[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])], 
                    self.smooth_region_local,self.unique,lambda_factor,self.link)
                    for chunk in range(0,len(self.fits),chunksize)])
                self.fits = list(chain.from_iterable(self.fits))
            else:
                self.fits = ray.get([self.smooth_region.remote(self.fits[reg_num],self.n_param_abd,self.genomic_positions[self.target_regions == region], np.nanmean(self.coverage[self.target_regions == region],axis=1), lambda_factor, self.link) 
                         for reg_num,region in enumerate(self.individual_regions)])
            ray.shutdown()
        else:
            self.fits = [self.smooth_region_local(self.fits[reg_num],param,self.param_names,self.n_param_abd,self.genomic_positions[self.target_regions == region], np.nanmean(self.coverage[self.target_regions == region],axis=1), lambda_factor, self.link) 
                     for reg_num,region in enumerate(self.individual_regions)]

    @staticmethod
    @ray.remote
    def smooth_chunk(region_id,fits,n_param_abd,genomic_positions,coverage,smooth_region,unique,lambda_factor,link="arcsin"):
        """
        Fit beta binomial model to a chunk of target regions from DNA methylation data.
    
        Parameters:
            chunk_regions (numpy array): Array of region assignments for this chunk.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
            maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
            maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs in this chunk. 
        """
        fits = [smooth_region(fits[reg_num],param,param_names,n_param_abd,genomic_positions[region_id==region],np.nanmean(coverage[region_id==region],axis=1),lambda_factor,link) 
                         for reg_num,region in enumerate(unique(region_id))]
        
        return fits
        
    @staticmethod
    def smooth_region_local(beta,param,param_names,n_par_abd,genomic_positions,cov,lambda_factor=10,link= "arcsin"):
        genomic_distances = np.abs(genomic_positions[:, None] - genomic_positions[None, :])
        
        if link == "arcsin":
            param = np.sin(beta[:,np.where(param_names==param)[0][0]]**2)
        elif link == "logit":
            param = expit(beta[:,np.where(param_names==param)[0][0]])
        
        # Compute pairwise differences in param
        param_diff = abs(param[:, None] - param[None, :])  
        
        # Compute kernel weights
        genomic_weights = 100**(0.001*-genomic_distances)
        param_weights = 1e20**(-param_diff) 
        
        # Combine weights
        combined_weights = genomic_weights * param_weights * 0.01/(0.01+1.5**-cov) * lambda_factor
        np.fill_diagonal(combined_weights,1)
        
        # Normalize weights for each CpG
        normalized_weights = combined_weights / combined_weights.sum(axis=1, keepdims=True)
        
        # Smooth abundance paramters
        beta = normalized_weights @ beta
        beta = normalized_weights @ beta

        return beta

    @staticmethod
    @ray.remote
    def smooth_region(beta, n_par_abd, genomic_positions, cov, lambda_factor=10, link= "arcsin"):
        genomic_distances = np.abs(genomic_positions[:, None] - genomic_positions[None, :])
        
        if link == "arcsin":
            mu = np.sin(beta[:,0]**2)
        elif link == "logit":
            mu = expit(beta[:,0])
        
        # Compute pairwise differences in mu
        mu_diff = abs(mu[:, None] - mu[None, :])  
        
        # Compute kernel weights
        genomic_weights = 100**(0.001*-genomic_distances)
        mu_weights = 1e20**(-mu_diff) 
        
        # Combine weights
        combined_weights = genomic_weights * mu_weights * 0.01/(0.01+1.5**-cov) * lambda_factor
        np.fill_diagonal(combined_weights,1)
        
        # Normalize weights for each CpG
        normalized_weights = combined_weights / combined_weights.sum(axis=1, keepdims=True)
        
        # Smooth abundance paramters
        beta = normalized_weights @ beta
        beta = normalized_weights @ beta
        return beta
    
    def wald_test(self, coef, padjust_method='fdr_bh', n_permute=1, find_dmrs='HMM', prop_sig=0.5, 
                  fdr_thresh=0.05, max_gap=1000, min_cpgs=3, n_states=3, state_labels=None, ncpu=1):
        if isinstance(coef, str):
            try:
                coef_idx = self.param_names_abd == coef
            except ValueError:
                raise ValueError(f"Can't find terms to be tested: {coef}. Make sure it matches a column name in design matrix.")

        # Hypothesis testing
        p = self.X.shape[1]
        betas = np.vstack(self.fits)[:,:self.n_param_abd][:, coef_idx].flatten()

        # Take out SE estimates 
        ses = np.vstack(self.se)[:,:self.n_param_abd][:, coef_idx].flatten()

        # Wald test, get p-values and FDR
        stat = betas / ses
        pvals = 2 * norm.sf(np.abs(stat))
        fdrs = multipletests(pvals, method=padjust_method)[1]

        # Results data frame
        res = pd.DataFrame({
            'chr': self.chr,
            'pos': self.genomic_positions,
            'stat': stat,
            'pvals': pvals,
            'fdrs': fdrs
        })

        if n_permute > 1:
            # Permute the labels of the specified column
            perm_stats = self.permute_and_refit(coef, N=n_permute, ncpu=ncpu)
            # Calculate the empirical p-value (two-sided)
            emp_pval = np.mean(perm_stats["stats"].values >= stat[:, np.newaxis], axis=1)

            res["emp_pvals"] = emp_pval
            res["emp_fdrs"] = multipletests(emp_pval, method=padjust_method)[1]
            
        if find_dmrs == 'binary_search':
            dmr_res = self.find_significant_regions(res,prop_sig=prop_sig,fdr_thresh=fdr_thresh,max_gap=max_gap)
            if n_permute > 1:
                permuted_dmrs = []
                for perm in range(n_permute):
                    # Create a mask that is True for all columns except the one to delete
                    stat = perm_stats["stats"].iloc[:, perm].values
                    emp_pval = np.mean(
                        perm_stats["stats"].iloc[:,-perm].values >= 
                        stat[:, np.newaxis], axis=1)
                    perm_res = pd.DataFrame({
                                            'chr': self.chr,
                                            'pos': self.genomic_positions,
                                            'stat': perm_stats["stats"].values[:,perm],
                                            'pvals': perm_stats["pvals"].values[:,perm],
                                            'fdrs': perm_stats["fdrs"].values[:,perm],
                                            'emp_pvals': emp_pval,
                                            'emp_fdrs': multipletests(emp_pval, method=padjust_method)[1]
                                        })
                    permuted_dmrs.append(self.find_significant_regions(perm_res,prop_sig=prop_sig,
                                                                       fdr_thresh=fdr_thresh,max_gap=max_gap,
                                                                       min_cpgs=min_cpgs))

                # Concatenate the permuted DMRs, if none find add one with 0 cpgs
                permuted_prop_sig_cpgs = [perm.prop_sig_cpgs.values if not perm.empty else np.array([0]) for perm in permuted_dmrs]
                permuted_prop_sig_cpgs = np.hstack(permuted_prop_sig_cpgs)
                # Calculate the empirical p-value - how many times did we see such a region in the permutations
                emp_pval = [np.mean(permuted_prop_sig_cpgs >= region_stats) for region_stats in dmr_res["prop_sig_cpgs"]]
                dmr_res["emp_pvals"] = emp_pval
                dmr_res["emp_fdrs"] = multipletests(emp_pval, method=padjust_method)[1]

            return res, dmr_res
        
        elif find_dmrs == 'HMM':
            dmr_res = self.find_significant_regions_HMM(res, n_states=n_states, min_cpgs=min_cpgs, fdr_thresh=fdr_thresh, 
                                                        prop_sig_thresh=prop_sig, state_labels=state_labels, 
                                                        hmm_plots=False, hmm_internals=False)
            return res, dmr_res

        else:
            return res
    
    @staticmethod
    def find_significant_regions(df, prop_sig=0.5, fdr_thresh=0.05, maxthresh=0.2, max_gap=1000, min_cpgs=3):
        """
        Find regions of adjacent CpGs where:
        - The gap between successive CpGs is less than max_gap,
        - The overall proportion of CpGs with fdr < fdr_thresh is >= prop_sig,
        - The last CpG in the region is significant (fdr < fdr_thresh),
        - And the region ends if a CpG has fdr > maxthresh.

        Parameters:
            df (pd.DataFrame): DataFrame with columns 'chr', 'pos', and 'fdrs'.
            max_gap (int): Maximum allowed gap between adjacent CpGs.
            prop_sig (float): Minimum proportion of CpGs in a region that must have fdr < fdr_thresh.
            fdr_thresh (float): FDR threshold for significance.
            maxthresh (float): FDR threshold above which a region will end.
            min_cpgs (int): Minimum number of CpGs required in a region.

        Returns:
            pd.DataFrame: DataFrame with columns 'chr', 'start', 'end', 'num_cpgs',
                        'num_sig_cpgs', and 'prop_sig_cpgs' for each region.
        """
        significant_regions = []
        # Sort the DataFrame by chromosome and position.
        df = df.sort_values(by=['chr', 'pos']).reset_index(drop=True)
        
        # Process each chromosome separately.
        for chr_name, group in df.groupby('chr'):
            # Reset index within the group for 0-based indexing.
            chr_df = group.reset_index(drop=True)
            positions = chr_df['pos'].to_numpy()
            # Binary significance indicator: 1 if fdr < fdr_thresh, 0 otherwise.
            sig = (chr_df['fdrs'] <= fdr_thresh).astype(int).to_numpy()
            n = len(chr_df)
            used_cpgs = set()  # For this chromosome only.

            i = 0
            while i < n:
                # Skip if not significant or already used.
                if sig[i] == 0 or i in used_cpgs:
                    i += 1
                    continue

                # Identify contiguous block where gaps are within max_gap.
                start_idx = i
                end_idx = i
                while end_idx + 1 < n and (positions[end_idx + 1] - positions[end_idx] <= max_gap):
                    # Stop the region if the FDR exceeds maxthresh.
                    if chr_df['fdrs'].iloc[end_idx + 1] > maxthresh:
                        break
                    end_idx += 1
                    if end_idx in used_cpgs:
                        break

                # Binary search over [i, end_idx] for the furthest index where
                # the overall proportion meets prop_sig.
                lo, hi = i, end_idx
                valid_idx = i
                while lo <= hi:
                    mid = (lo + hi) // 2
                    num_cpgs = mid - i + 1
                    num_sig = int(np.sum(sig[i:mid+1]))
                    if num_sig / num_cpgs >= prop_sig:
                        valid_idx = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                
                # Now ensure that the last CpG in the region is itself significant.
                while valid_idx > i and chr_df['fdrs'].iloc[valid_idx] >= fdr_thresh:
                    valid_idx -= 1

                # Define the region using the trimmed indices.
                region = chr_df.iloc[i:valid_idx + 1]
                num_cpgs = len(region)
                num_sig = (region['fdrs'] < fdr_thresh).sum()
                prop = num_sig / num_cpgs if num_cpgs > 0 else 0
                
                # Only record if region still meets criteria.
                if num_cpgs >= min_cpgs and prop >= prop_sig:
                    significant_regions.append({
                        'chr': chr_name,
                        'start': region['pos'].iloc[0],
                        'end': region['pos'].iloc[-1],
                        'num_cpgs': num_cpgs,
                        'num_sig_cpgs': num_sig,
                        'prop_sig_cpgs': prop
                    })
                    used_cpgs.update(region.index)
                
                # Move to the next candidate region.
                i = valid_idx + 1

        return pd.DataFrame(significant_regions)
    
    @staticmethod
    def find_significant_regions_HMM(cpg_res, n_states=3, min_cpgs=5, fdr_thresh=0.05, prop_sig_thresh=0.5, 
                                 state_labels=None, hmm_plots=False, hmm_internals=False):
        """
        Identify candidate DMR regions using an HMM with multiple states and multivariate features.
        
        The function uses two observed features per CpG site: for instance, 
        'score1' (e.g., –log10(p-value)) and 'score2' (e.g., a test statistic). 
        An HMM is applied per chromosome to decode the state at each CpG; contiguous CpGs 
        in a non-background state (e.g., hyper- or hypomethylated) are merged into candidate regions.
        
        Parameters
        ----------
        cpg_res : pd.DataFrame
            Must contain the following columns:
            - 'chr': Chromosome identifier
            - 'pos': Genomic position
            - 'pvals': CpG pvalue
            - 'fdrs': CpG fdrs
            - 'stat': CpG Test Statistic
        n_states : int, default 3
            Number of HMM states. Here we assume:
            0: Background,
            1: Hypermethylated DMR,
            2: Hypomethylated DMR.
        min_cpgs : int, default 3
            Minimum number of consecutive CpGs to report a region.
        state_labels : dict or None
            Optional mapping from state index to state name. If None, defaults are:
            {0: 'Background', 1: 'Hypermethylated', 2: 'Hypomethylated'}.
        hmm_plots : bool, default False
            Whether to generate plots for HMM state decoding.
        hmm_internals : bool, default False
            Whether to print internal HMM parameters (e.g., means, transition probabilities).
        
        Returns
        -------
        regions_df : pd.DataFrame
            Candidate regions with columns:
            - 'chr': Chromosome
            - 'start': Starting position of the region (first CpG)
            - 'end': Ending position of the region (last CpG)
            - 'num_cpgs': Number of CpGs in the region
            - 'state': The region state ('Hypermethylated' or 'Hypomethylated')
        """
        if state_labels is None:
            state_labels = {0: 'Background', 1: 'Hypermethylated', 2: 'Hypomethylated'}
        
        candidate_regions = []

        cpg_res['-log10pval'] = -np.log10(cpg_res['pvals'])
        
        # Process each chromosome separately.
        for chrom, group in cpg_res.groupby('chr'):
            group = group.sort_values('pos').reset_index(drop=True)
            # Create observations array with both score1 and score2.
            obs = group[['-log10pval', 'stat']].values  # shape: (n_samples, 2)
            
            # Initialize a Gaussian HMM with three states and diagonal covariance.
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42, init_params="c")

            model.transmat_ = np.array([
                [0.9999, 0.00005, 0.00005],  # From background
                [0.005, 0.99, 0.005],          # From hyper
                [0.005, 0.005, 0.99]           # From hypo
            ])
            
            # Initialize means. Adjust these based on your data.
            # Background state: score1 ~ 0.5 (low significance), score2 near 0.
            # Hypermethylated: high score1 and positive test statistic.
            # Hypomethylated: high score1 and negative test statistic.
            model.means_ = np.array([
                [0, 0.0],    # State 0: Background 
                [3.0, 3.0],   # State 1: Hypermethylated 
                [3.0, -3.0]   # State 2: Hypomethylated
            ])
            # Initialize diagonal covariances; these can be tuned:
            model.covars_ = np.tile(np.array([1.0, 1.0]), (n_states, 1))
            
            # Fit the HMM to the observations using Baum-Welch:
            model.fit(obs)
            
            # Decode the most likely state sequence:
            state_seq = model.predict(obs)
            group['state'] = state_seq
            
            # (Optional) Plot for inspection:
            if hmm_plots:
                plt.figure(figsize=(10, 4))
                plt.plot(group['pos'], group['-log10pval'], 'o-', label='score1 (-log10 pvalue)')
                plt.plot(group['pos'], group['stat'], 'o-', label='score2 (test statistic)', alpha=0.7)
                plt.step(group['pos'], state_seq, where='mid', label='Decoded State', color='black')
                plt.xlabel('Genomic Position')
                plt.ylabel('Score / State')
                plt.legend()
                plt.title(f"Chromosome {chrom} Observations and Decoded State")
                plt.show()

            if hmm_internals:
                print("hmm state means:\n", model.means_)
                print("hmm state transition probabilities:\n", model.transmat_)

            # Extract contiguous segments that are in a DMR state (states 1 and 2).
            current_region = None
            for i, row in group.iterrows():
                st = row['state']
                if st != 0:  # non-background state
                    if current_region is None:
                        current_region = {
                            'chr': chrom,
                            'start_idx': i,
                            'end_idx': i,
                            'state': st  # record the state for current contiguous block
                        }
                    else:
                        if st == current_region['state']:
                            current_region['end_idx'] = i
                        else:
                            # End the current region and start a new one if it's long enough.
                            if (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                                start_pos = group.loc[current_region['start_idx'], 'pos']
                                end_pos = group.loc[current_region['end_idx'], 'pos']
                                candidate_regions.append({
                                    'chr': chrom,
                                    'start': start_pos,
                                    'end': end_pos,
                                    'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                                    'num_sig': (cpg_res.loc[(cpg_res.chr == chrom) &
                                                            (cpg_res.pos >= start_pos) &
                                                            (cpg_res.pos <= end_pos), 'fdrs'] <= fdr_thresh).sum(),
                                    'state': state_labels[current_region['state']]
                                })
                            current_region = {
                                'chr': chrom,
                                'start_idx': i,
                                'end_idx': i,
                                'state': st
                            }
                else:
                    if current_region is not None:
                        if (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                            start_pos = group.loc[current_region['start_idx'], 'pos']
                            end_pos = group.loc[current_region['end_idx'], 'pos']
                            candidate_regions.append({
                                'chr': chrom,
                                'start': start_pos,
                                'end': end_pos,
                                'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                                'num_sig': (cpg_res.loc[(cpg_res.chr == chrom) &
                                                        (cpg_res.pos >= start_pos) &
                                                        (cpg_res.pos <= end_pos), 'fdrs'] <= fdr_thresh).sum(),
                                'state': state_labels[current_region['state']]
                            })
                        current_region = None

            # Check if a region remains at the end:
            if current_region is not None and (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                start_pos = group.loc[current_region['start_idx'], 'pos']
                end_pos = group.loc[current_region['end_idx'], 'pos']
                candidate_regions.append({
                    'chr': chrom,
                    'start': start_pos,
                    'end': end_pos,
                    'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                    'num_sig': (cpg_res.loc[(cpg_res.chr == chrom) &
                                            (cpg_res.pos >= start_pos) &
                                            (cpg_res.pos <= end_pos), 'fdrs'] <= fdr_thresh).sum(),
                    'state': state_labels[current_region['state']]
                })
        
        region_res = pd.DataFrame(candidate_regions)
        region_res['prop_sig'] = region_res['num_sig'] / region_res['num_cpgs']
        region_res = region_res[region_res['prop_sig'] >= prop_sig_thresh]
        return region_res

    def permute_and_refit(self, coef, N=100, padjust_method='fdr_bh', ncpu=1):
        """
        Refit the beta-binomial regression model and compute Wald test statistics N times,
        randomly permuting the labels of a given column in self.X.

        Parameters:
            column (str): The name of the column in self.X to permute.
            N (int): The number of permutations.
            padjust_method (str): Method to adjust p-values for multiple testing.

        Returns:
            pd.DataFrame: DataFrame containing the permuted Wald test statistics.
        """
        assert coef in self.param_names_abd, f"Column '{coef}' not found in self.X"
        coef_idx = self.param_names_abd == coef
        permuted_stats = []
        permuted_pvals = []
        permuted_fdrs = []

        for i in range(N):
            # Permute the labels of the specified column
            permuted_obj = self.copy()
            permuted_obj.X[:,coef_idx] = np.random.permutation(permuted_obj.X[:,coef_idx])

            # Refit the model
            permuted_obj.fit_betabinom(ncpu=ncpu)

            # Compute Wald test statistics
            wald_res = permuted_obj.wald_test(coef, padjust_method=padjust_method, n_permute=1, find_dmrs=False)
            permuted_stats.append(wald_res['stat'].values)
            permuted_pvals.append(wald_res['pvals'].values)
            permuted_fdrs.append(wald_res['fdrs'].values)

        # Combine results into a DataFrame
        permuted_stats_df = pd.DataFrame(np.vstack(permuted_stats).T, columns=[f'perm_{i}' for i in range(N)])
        permuted_pvals_df = pd.DataFrame(np.vstack(permuted_pvals).T, columns=[f'perm_{i}' for i in range(N)])
        permuted_fdrs_df = pd.DataFrame(np.vstack(permuted_fdrs).T, columns=[f'perm_{i}' for i in range(N)])

        permuted_stats = {"stats": permuted_stats_df, "pvals": permuted_pvals_df, "fdrs": permuted_fdrs_df}

        return permuted_stats

    def find_codistrib_regions(self,site_names:np.ndarray=np.array([]),dmrs: bool=True,min_cpgs: int=3,ncpu: int=1,maxiter=500,maxfev=500,chunksize=1):
        """
        Perform differential methylation analysis using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
        mean and association with covariates are similar.
        
        Parameters:
            site_names (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying cpg name. If not provided names each cpg by its index.
            dmrs (boolean, default: True): Whether to identify and test differentially methylated regions of contigous cpgs whose mean and association with covariates are similar.
            min_cpgs (integer, default: 3): Minimum number of cpgs comprising a dmr for it to be retained.
            ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
            
        Returns:
            pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
        """
        
        assert any([len(site_names) == self.ncpgs,len(site_names) == 0]), "length of site_names must be equal to the number of cpgs"
        assert isinstance(min_cpgs,int), "min_cpgs must be positive integer"
        assert ncpu>0, "min_cpgs must be positive integer"
        assert isinstance(ncpu,int), "ncpu must be positive integer"
        assert ncpu>0, "ncpu must be positive integer"
        
        if len(site_names) == 0:
            site_names = list(range(self.ncpgs))
        
        if ncpu > 1:
            ray.init(num_cpus=ncpu)
        
            if chunksize > 1:
                codistrib_regions = ray.get([self.find_codistrib_chunk.remote(
                    self.target_regions[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.find_codistrib_local,self.beta_binomial_log_likelihood,self.unique,
                    self.fits[chunk:chunk+chunksize],
                    self.meth[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.coverage[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.X,self.X_star,min_cpgs,maxiter,maxfev,self.param_names_abd,self.param_names_disp,self.link)
                    for chunk in range(0,len(set(self.individual_regions))+1,chunksize)])
            else:
                codistrib_regions = ray.get([self.find_codistrib.remote(region,self.beta_binomial_log_likelihood,self.fits[fit],self.meth[self.target_regions==region],self.coverage[self.target_regions==region],
                                                                                self.X,self.X_star,min_cpgs,self.param_names_abd,self.param_names_disp,maxiter,maxfev,self.link) 
                                                    for fit,region in enumerate(self.individual_regions)])
            ray.shutdown()
            
            self.codistrib_regions = np.hstack(codistrib_regions)
        
        else:
            
            codistrib_regions = [self.find_codistrib_local(region,self.beta_binomial_log_likelihood,self.fits[fit],self.meth[self.target_regions==region],
                                                            self.coverage[self.target_regions==region],self.X,self.X_star,min_cpgs,
                                                            self.param_names_abd,self.param_names_disp,maxiter,maxfev,self.link) 
                                                            for fit,region in enumerate(self.individual_regions)]
                    
            self.codistrib_regions = np.hstack(codistrib_regions)

    @staticmethod
    @ray.remote
    def find_codistrib_chunk(region_id,find_codistrib,beta_binomial_log_likelihood,unique,fits,meth,coverage,X_id,X_star_id,min_cpgs=3,maxiter=150,maxfev=150,param_names_abd=["intercept"],param_names_disp=["intercept"],link="arcsin"):
        """
        Fit beta binomial model to a chunk of target regions from DNA methylation data.
    
        Parameters:
            chunk_regions (numpy array): Array of region assignments for this chunk.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
            maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
            maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs in this chunk. 
        """
        codistrib_regions = [find_codistrib(region,beta_binomial_log_likelihood,fits[fit],
                                                 meth[region_id==region],coverage[region_id==region],
                                                 X_id,X_star_id,min_cpgs,param_names_abd,param_names_disp,
                                                 maxiter,maxfev,link) 
                                                   for fit,region in enumerate(unique(region_id))]
        
        codistrib_regions = np.hstack(codistrib_regions)
        
        return codistrib_regions


    @staticmethod
    def find_codistrib_local(region,beta_binomial_log_likelihood,fits,meth,coverage,X,X_star,min_cpgs=3,param=["intercept"],param_names_abd=["intercept"],
                              param_names_disp=["intercept"],maxiter=500,maxfev=500,link="arcsin"):
        """
        Computes differentially methylated regions of contigous cpgs whose mean, dispersion, and association with covariates are similar.
    
        Parameters:
            region (integer, float, or string): Name of the region being analysed.
            fits (1D numpy array): Array containing bb fits (corcon_2 object) for all cpgs in the region.
            min_cpgs (integer, default: 3): Minimum length of a dmr.
            
        Returns:
            pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
        """  
        start=0
        end=2
        codistrib_regions=np.array([])
        n_params_abd=X.shape[1]
        n_params_disp=X_star.shape[1]
        n_samples = X.shape[0]
        n_params=2
        bad_cpgs=0    
        X=X[:,[0]]
        X_c=X[:,[0]]
        X_star_c=X_star[:,[0]]
        fits = fits[:,[0,n_params_abd]] # Subset fits to only intercepts (mean abundance and dispersion)
        n_params_abd=1
        n_params_disp=1
        ll_c = beta_binomial_log_likelihood(meth[start],coverage[start],X,X_star,fits[start],n_params_abd,n_params_disp,link=link)
        
        while end < len(fits)+1:
            
            if start+min_cpgs < len(fits)+1: #Can we form a codistrib region
                X_c=np.vstack([X_c,X])
                X_star_c=np.vstack([X_star_c,X_star])
                ll_s = ll_c + beta_binomial_log_likelihood(meth[end-1],coverage[end-1],X,X_star,fits[end-1],
                                                           n_params_abd,n_params_disp,link=link)
                ll_c = beta_binomial_log_likelihood(meth[start:end].flatten(),coverage[start:end].flatten(),X_c,X_star_c,
                                                    fits[start],n_params_abd,n_params_disp,link=link) 
    
                bic_c = -2 * ll_c + (n_params) * np.log(n_samples) 
                bic_s = -2 * ll_s + (n_params*3) * np.log(n_samples) 
                end += 1
    
        #If sites come from the same distribution, keep extending the region, else start a new region
                if any([all([ll_c<0, bic_c > bic_s]),all([ll_c>0, bic_c < bic_s])]):
                    if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                        codistrib_regions=np.hstack([codistrib_regions,np.repeat(f'{region}_{start}-{end-3}', end-(start+2))])
                    else:
                        codistrib_regions=np.hstack([codistrib_regions,np.array([f'{int(cpg)}' for cpg in range(start,end-2)])])
                    start=end-2
                    X_c=X
                    X_star_c=X_star
                    ll_c=beta_binomial_log_likelihood(meth[start],coverage[start],X,X_star,fits[start],n_params_abd,
                                                      n_params_disp,link=link)
                    
                else:
                    ll_s=ll_c             
    
            else:
                end += 1
           
    
        if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
            codistrib_regions=np.hstack([codistrib_regions,np.repeat(f'{region}_{start}-{end}', end-(start+1))])
        else:
            codistrib_regions=np.hstack([codistrib_regions,np.array([f'{int(cpg)}' for cpg in range(start,len(fits))])])
            
        return codistrib_regions

    @staticmethod
    @ray.remote
    def find_codistrib(region,beta_binomial_log_likelihood,fits,meth,coverage,X,X_star,min_cpgs=3,param=["intercept"],param_names_abd=["intercept"],
                              param_names_disp=["intercept"],maxiter=500,maxfev=500,link="arcsin"):
        """
        Computes differentially methylated regions of contigous cpgs whose mean, dispersion, and association with covariates are similar.
    
        Parameters:
            region (integer, float, or string): Name of the region being analysed.
            fits (1D numpy array): Array containing bb fits (corcon_2 object) for all cpgs in the region.
            min_cpgs (integer, default: 3): Minimum length of a dmr.
            
        Returns:
            pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
        """  
        start=0
        end=2
        codistrib_regions=np.array([])
        n_params_abd=X.shape[1]
        n_params_disp=X_star.shape[1]
        n_samples = X.shape[0]
        n_params=2
        bad_cpgs=0    
        X=X[:,[0]]
        X_c=X[:,[0]]
        X_star_c=X_star[:,[0]]
        fits = fits[:,[0,n_params_abd]] # Subset fits to only intercepts (mean abundance and dispersion)
        n_params_abd=1
        n_params_disp=1
        ll_c = beta_binomial_log_likelihood(meth[start],coverage[start], X,X_star,fits[start],n_params_abd,n_params_disp,link=link)
        
        while end < len(fits)+1:
            
            if start+min_cpgs < len(fits)+1: #Can we form a codistrib region
                X_c=np.vstack([X_c,X])
                X_star_c=np.vstack([X_star_c,X_star])
                ll_s = ll_c + beta_binomial_log_likelihood(meth[end-1],coverage[end-1],X,X_star,fits[end-1],
                                                           n_params_abd,n_params_disp,link=link)
                ll_c = beta_binomial_log_likelihood(meth[start:end].flatten(),coverage[start:end].flatten(),X_c,X_star_c,
                                                    fits[start],n_params_abd,n_params_disp,link=link) 
    
                bic_c = -2 * ll_c + (n_params) * np.log(n_samples) 
                bic_s = -2 * ll_s + (n_params*3) * np.log(n_samples) 
                end += 1
    
        #If sites come from the same distribution, keep extending the region, else start a new region
                if any([all([ll_c<0, bic_c > bic_s]),all([ll_c>0, bic_c < bic_s])]):
                    if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                        codistrib_regions=np.hstack([codistrib_regions,np.repeat(f'{region}_{start}-{end-3}', end-(start+2))])
                    else:
                        codistrib_regions=np.hstack([codistrib_regions,np.array([f'{int(cpg)}' for cpg in range(start,end-2)])])
                    start=end-2
                    X_c=X
                    X_star_c=X_star
                    ll_c=beta_binomial_log_likelihood(meth[start],coverage[start], X,X_star,fits[start],n_params_abd,
                                                      n_params_disp,link=link)
                    
                else:
                    ll_s=ll_c             
    
            else:
                end += 1
           
    
        if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
            codistrib_regions=np.hstack([codistrib_regions,np.repeat(f'{region}_{start}-{end}', end-(start+1))])
        else:
            codistrib_regions=np.hstack([codistrib_regions,np.array([f'{int(cpg)}' for cpg in range(start,len(fits))])])
            
        return codistrib_regions
    
    @staticmethod
    def beta_binomial_log_likelihood(count,total,X,X_star,beta,n_param_abd,n_param_disp,
                                 link="arcsin",max_param=1e+10):
        """
        Compute the negative log-likelihood for beta-binomial regression.
    
        Parameters:
            beta (numpy array): Coefficients to estimate (p-dimensional vector).
            
        Returns:
            float: Negative log-likelihood.
        """
        #Remove nans from calculation
        X=X[~np.isnan(count)]
        X_star=X_star[~np.isnan(count)]
        total=total[~np.isnan(count)]
        count=count[~np.isnan(count)]

        # Reshape beta into segments for mu and phi
        beta_mu = beta[:n_param_abd]
        beta_phi = beta[-n_param_disp:]
        
        # Compute linear predictors
        mu_wlink = X @ beta_mu.T
        phi_wlink = X_star @ beta_phi.T
    
        # Transform to scale (0, 1)
        if link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif link == "logit":
            mu = expit(mu_wlink)
            phi = expit(phi_wlink)
    
        # Precompute shared terms
        phi_inv = (1 - phi) / phi
        a = mu * phi_inv
        b = (1 - mu) * phi_inv

        # Scale parameters to avoid numerical issues
        scale_factor = np.maximum(a, b) / max_param
        scale_factor = np.maximum(scale_factor, 1)  # Ensure scale_factor is at least 1
        a /= scale_factor
        b /= scale_factor
    
        # Compute log-likelihood in a vectorized manner
        log_likelihood = np.sum(stats.betabinom.logpmf(count, total, a.T, b.T))

        return log_likelihood

    @njit
    def find_subregions(self, coef,  tol=0.05, max_dist=10000, start_group=0):
        """
        Return an array of segment labels for x such that:
        - Normally, a new segment is started when the value differs from the current baseline or 
            the previous value by more than tol.
        - If a single value exceeds tol but the immediately following value is within tol of the current baseline,
            the outlier is given a group on its own and the segment continues.
        
        Parameters
        ----------
        coef : Name of coeficient to find subregions for (in self.param_names_abd)
        tol : numeric threshold for allowed deviation from the group's baseline and previous value.
        
        Returns
        -------
        labels : 1D numpy array of int64 group labels.
        """
        if isinstance(coef, str):
            try:
                coef_idx = self.param_names_abd == coef
            except ValueError:
                raise ValueError(f"Can't find coef: {coef}. Make sure it matches a column name in design matrix.")

        # Extract coef values
        x = np.vstack(self.fits)[:,:self.n_param_abd][:, coef_idx].flatten()
        labels = np.repeat(-1, self.ncpgs).astype(np.int64)
        group = start_group
        baseline = x[0]
        labels[0] = group
        outliers=0
        for i in range(1, n):
            if labels[i] != -1: # Skip if already marked as part of a group.
                continue
            if (self.chr[i] != self.chr[i-1]) or (self.genomic_positions[i] - self.genomic_positions[i-1] > max_dist): # Start new group if new chromosome or more than max dist apart
                group += 1+outliers
                baseline = x[i]
                labels[i] = group
                outliers=0
                continue
            # Check if current value is within tol of the current baseline.
            if (np.abs(x[i] - baseline) <= tol) and (np.abs(x[i] - x[i-1]) <= tol):
                labels[i] = group
            else:
                # The current value is "far" from the baseline.
                # Check if it's an isolated outlier: if the next value exists and is within tol of the baseline.
                if (i < self.ncpgs - 1) and (np.abs(x[i+1] - baseline) <= tol) and (np.abs(x[i+1] - x[i-1]) <= tol):
                    # Mark this value as an isolated outlier.
                    labels[i] = group+1+outliers
                    labels[i+1] = group
                    outliers+=1
                    # Do not update baseline or group.
                else:
                    # Otherwise, treat this as a true group break:
                    group += 1+outliers
                    baseline = x[i]
                    labels[i] = group
                    outliers=0
        return labels

    def sim_multiple_cpgs(self,covs=None,covs_disp=None,use_codistrib_regions: bool=True,read_depth: str|int|np.ndarray="from_data",vary_read_depth=True,read_depth_sd: str|int|float|np.ndarray="from_data",
                          adjust_factor: float|list=0, diff_regions_up: list|np.ndarray=[],diff_regions_down: list|np.ndarray=[],n_diff_regions: list|int=0,prop_pos: float=0.5,
                          sample_size: int=100,ncpu: int=1,chunksize=1):
        """
        Simulate new samples based on existing samples.
    
        Parameters:
            covs (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the simulated samples for the 
            mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form an intercept column, and level 0 for all other covariates.
            covs_disp (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            If not supplied will form an intercept column, and level 0 for all other covariates.
            use_codistrib_regions (bool, default: True): Whether to adjust whole codistributed regions (True) or single cpgs (False) by adjust_factor. Need to run bbseq()
            to set True.
            read_depth (integer or "from_data", default: "from_data"): Desired average read depth of simulated samples. If "from_data" calculates average read depth per region
            from object data, and uses this as read_depth for simulations.
            vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
            read_depth_sd (integer or "from_data", default: "from_data"): Desired average standard deviation read depth between simulated samples. If "from_data" calculates average SD of read depth 
            per region between samples in object data, and uses this as read_depth_sd for simulations.
            adjust_factor (float 0-1, default: 0): The percent methylation 'diff_regions' or 'n_diff_regions' will be altered by in simulated samples.
            diff_regions_up (List or numpy array, default: []): Names of regions whose probability of methylation will be increased by 'adjust_factor'.
            diff_regions_down (List or numpy array, default: []): Names of regions whose probability of methylation will be decreased by 'adjust_factor'.
            n_diff_regions (integer or list, default: 0): If diff_regions not specified, the number of regions to be affected by 'adjust_factor'. Can be a list of length 2 specifying number regions to increase
            or decrease in methylation.
            and ne
            prop_pos (Float 0-1, default: 0.5): Which proportion of adjusted regions should be increased in methylation, versus decreased.
            sample_size (integer, default: 100): Number of samples to simulate.
            ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
            
        Returns:
            2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
        """

        assert len(self.fits) == len(self.individual_regions), "Run fit before simulating"
        if not isinstance(read_depth,np.ndarray):
            assert any([read_depth == "from_data", isinstance(read_depth,(list,int))]), "read_depth must be 'from_data' or a positive integer"
        if isinstance(read_depth,(list)):
            assert all(read_depth>0) , "read_depth must be a positive integer"
        if isinstance(read_depth,(int)):
            assert read_depth>0, "read_depth must be a positive integer"
        if not isinstance(read_depth,np.ndarray):
            assert any([read_depth_sd == "from_data", isinstance(read_depth_sd,(list,float,int))]), "read_depth_sd must be 'from_data' or a positive integer or float"
        if isinstance(read_depth_sd,(list,int)):
            assert all(read_depth_sd>0), "read_depth_sd must be positive"
        if isinstance(adjust_factor,float):
            assert all([adjust_factor <= 1, adjust_factor>=0]), "adjust_factor must be a float or list of floats of length 2 between 0 and 1"
        if isinstance(adjust_factor,list):
            assert all([adjust_factor < [1,1], adjust_factor > [0,0]]), "adjust_factor must be a float or list of floats of length 2 between 0 and 1"
        assert isinstance(ncpu,int), "ncpu must be positive integer"
        assert ncpu>0, "ncpu must be positive integer"

        if isinstance(read_depth,str):
            if len(self.region_cov) == 0:
                self.region_cov = np.array([np.nanmean(region) for region in np.split(self.coverage,self.new_region_indices)])
            read_depth = self.region_cov
        elif isinstance(read_depth,int):
            read_depth = np.repeat(read_depth,len(self.individual_regions))
        elif isinstance(read_depth,np.ndarray):
            read_depth = np.tile(read_depth,(len(self.individual_regions),1)) # If numpy array repeat as 2d array for each region

        if isinstance(read_depth_sd,str):
            if len(self.region_cov_sd) == 0:
                self.region_cov_sd = np.array([np.nanstd(region) for region in np.split(self.coverage,self.new_region_indices)]) 
            read_depth_sd = self.region_cov_sd
        elif isinstance(read_depth_sd,(int,float)):
            read_depth_sd = np.repeat(read_depth_sd,len(self.individual_regions))
        elif isinstance(read_depth,np.ndarray):
            read_depth_sd = np.tile(read_depth_sd,(len(self.individual_regions),1)) # If numpy array repeat as 2d array for each region

        if not isinstance(covs, pd.DataFrame):
            if isinstance(covs, np.ndarray):
                if covs.shape[1] < self.n_param_abd: # Add intercept column if required
                    covs = pd.DataFrame(np.vstack([np.tile(np.repeat(1,sample_size),(1,1)),covs.T]).T)
                else:
                    covs = pd.DataFrame(covs)
            else:
                covs = pd.DataFrame(np.vstack([np.tile(np.repeat(1,sample_size),(1,1)), np.tile(np.repeat(0,sample_size),(self.n_param_abd-1,1))]).T)
            covs.columns=self.param_abd
            
        if not isinstance(covs_disp, pd.DataFrame):
            covs_disp = pd.DataFrame(np.vstack([np.tile(np.repeat(1,sample_size),(1,1)), np.tile(np.repeat(0,sample_size),(self.n_param_disp-1,1))]).T)
            covs_disp.columns=self.param_disp

        if isinstance(adjust_factor,(float,int)):
            adj_pos=adj_neg=adjust_factor
        else:
            adj_pos,adj_neg=adjust_factor

        if isinstance(n_diff_regions,int):
            n_pos=int(n_diff_regions*prop_pos)
            n_neg=int(n_diff_regions*(1-prop_pos))
        else:
            n_pos,n_neg=n_diff_regions

        adjust_factors = np.repeat(0.0, self.ncpgs)
        if use_codistrib_regions:
            assert len(self.codistrib_regions) == self.ncpgs, "Run bbseq before simulating if codistrib_regions=True"
            if len(diff_regions_up)>0 or len(diff_regions_down)>0 or n_pos>0 or n_neg>0:
                if all([len(diff_regions_up)==0,len(diff_regions_down)==0, any([n_pos>0 or n_neg>0])]):
                    self.codistrib_regions_only = np.array([region for region in self.unique(self.codistrib_regions) if "_" in region])
                    if isinstance(self.codistrib_region_min_beta,list):
                        self.is_codistrib_region = np.isin(self.codistrib_regions,self.codistrib_regions_only)
                        self.codistrib_region_min_beta,self.codistrib_region_max_beta = zip(*[self.min_max(region) for region in self.codistrib_regions_only])
                        self.codistrib_region_min_beta = np.array(self.codistrib_region_min_beta)
                        self.codistrib_region_max_beta = np.array(self.codistrib_region_max_beta)
                    if isinstance(self.disp_intercept,list):
                        self.disp_intercept = np.array([self.min_max(region,"disp","max") for region in self.codistrib_regions_only])
                    try:
                        diff_regions_up = np.random.choice(self.codistrib_regions_only[(self.codistrib_region_max_beta<=0.5) & (self.codistrib_region_max_beta+adj_pos < 1) & (self.disp_intercept < 0.1)],n_pos)
                    except:
                        try:
                            diff_regions_up = np.random.choice(self.codistrib_regions_only[self.codistrib_region_max_beta+adj_pos < 0.99],n_pos)
                        except:
                            raise ValueError("Less than n_diff_regions regions with low enough probability methylation to be raised by adjust_factor without going above 100% methylated. Likely adjust_factor or n_diff_regions is too high.")
                    try:
                        diff_regions_down = np.random.choice(self.codistrib_regions_only[(self.codistrib_region_min_beta>=0.5) & (self.codistrib_region_min_beta-adj_neg > 0) & (self.disp_intercept < 0.1)
                                                             & ~np.isin(self.codistrib_regions_only,diff_regions_up)],n_neg)
                    except:
                        try:
                            diff_regions_down = np.random.choice(self.codistrib_regions_only[(self.codistrib_region_min_beta-adj_neg > 0.01) & ~np.isin(self.codistrib_regions_only,diff_regions_up) & 
                                                                 (self.codistrib_region_min_beta-adj_neg > 0)],n_neg)
                        except:
                            raise ValueError("Less than n_diff_regions regions with high enough probability methylation to be lowered by adjust_factor without going below 0% methylated. Likely adjust_factor is or n_diff_regions is too high.")
                adjust_factors[np.isin(self.codistrib_regions,diff_regions_up)] = adj_pos
                adjust_factors[np.isin(self.codistrib_regions,diff_regions_down)] = -adj_neg

        elif adj_pos or adj_neg > 0:
            cpgs = np.array(range(self.ncpgs))
            sites_up = np.random.choice(cpgs,n_pos)
            sites_down = np.random.choice(cpgs[~np.isin(cpgs,sites_up)],n_neg)
            adjust_factors[np.isin(cpgs,sites_up)] = adj_pos
            adjust_factors[np.isin(cpgs,sites_down)] = -adj_neg

        adjust_factors = np.split(adjust_factors,self.new_region_indices)
        
        if ncpu > 1: #Use ray for parallel processing
            ray.init(num_cpus=ncpu)    
            sim_meth, sim_coverage = zip(*ray.get([self.sim_chunk.remote(
                self.individual_regions[chunk:chunk+chunksize],
                self.fits[chunk:chunk+chunksize],covs,covs_disp,self.sim_internal,read_depth[chunk:chunk+chunksize],
                vary_read_depth,read_depth_sd[chunk:chunk+chunksize],adjust_factors[chunk:chunk+chunksize],
                sample_size,self.link
            ) for chunk in range(0,len(set(self.individual_regions))+1,chunksize)]))
            ray.shutdown()

        else:
            sim_meth, sim_coverage = zip(*[self.sim_local(region_num,covs,covs_disp,read_depth[region_num],vary_read_depth,read_depth_sd[region_num],
                                                     adjust_factors[region_num],sample_size) for region_num in range(len(self.individual_regions))])
        if isinstance(adjust_factor,(int)):
            if adjust_factor == 0:
                return np.vstack(sim_meth), np.vstack(sim_coverage)
        return np.vstack(sim_meth),np.vstack(sim_coverage),np.hstack(adjust_factors),[diff_regions_up,diff_regions_down]

    @staticmethod
    @ray.remote
    def sim_chunk(region_id,fits,covs,covs_disp,sim_local,read_depth,vary_read_depth,read_depth_sd,adjust_factors,sample_size=100,link="arcsin"):
        """
        Fit beta binomial model to a chunk of target regions from DNA methylation data.
    
        Parameters:
            chunk_regions (numpy array): Array of region assignments for this chunk.
            meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
            region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
            maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
            maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs in this chunk. 
        """
        sim_meth, sim_coverage = zip(*[sim_local(fits[region_num],covs,covs_disp,
                                                          read_depth[region_num],vary_read_depth,read_depth_sd[region_num],
                                                     adjust_factors[region_num],sample_size,link) for region_num in range(len(region_id))])
        
        sim_meth=np.vstack(sim_meth)
        sim_coverage=np.vstack(sim_coverage)
        
        return sim_meth,sim_coverage

    def sim_local(self,region_num,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100):
        """
        Simulate new samples based on existing samples, for this region.
    
        Parameters:
            params (numpy array): Array of paramether estimates (columns) for each cpg in the region (rows). 
            X (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the simulated samples for the 
            mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            X_star (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            read_depth (integer, default: 30): Desired average read depth of simulated samples.
            vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
            read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
            adjust_factors (1D numpy array, default: 0): Array of the percent methylation each cpg will be altered by in the simulated versus template samples.
            sample_size (integer, default: 100): Number of samples to simulate.
            link (string, default: "arcsin"): Link function to use, either 'arcsin' or 'logit'.
            
        Returns:
            2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
        """
        params=self.fits[region_num]
        if vary_read_depth: # Create array of sample read depths, varying read depth if specified to
            read_depth = np.random.normal(read_depth,read_depth_sd,sample_size).astype(int)
            coverage = np.tile(read_depth,(params.shape[0],1))    
        else:
            coverage=np.empty((params.shape[0],sample_size),dtype=int)
            coverage.fill(read_depth)
        coverage=np.clip(coverage, a_min=1,a_max=None) # Ensure at least 1 coverage
        
        mu_wlink = np.matmul(
                X,
                params[:,:self.n_param_abd].T
            )
        phi_wlink = np.matmul(
                X_star,
                params[:,self.n_param_abd:].T
            )
        if self.link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif self.link == "logit":
            mu = expit(mu_wlink)
            phi = expit(phi_wlink)
            
        if any(adjust_factors != 0):
            mu=mu+adjust_factors
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        else:
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi

        # Correct any out of bounds parameters
        a[a <= 0] = 1
        b[b <= 0] = 1

        # try:
        meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 
        # except:
            #raise ValueError("Parameters are not compatible with the beta binomial model. Likely adjust_factor is too high or something went wrong during fitting.")
        return meth, coverage

    @staticmethod
    @ray.remote
    def sim(params,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100,link="arcsin"):
        """
        Simulate new samples based on existing samples, for this region.
    
        Parameters:
            params (numpy array): Array of paramether estimates (columns) for each cpg in the region (rows). 
            X (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the simulated samples for the 
            mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            X_star (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            read_depth (integer, default: 30): Desired average read depth of simulated samples.
            vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
            read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
            adjust_factors (1D numpy array, default: 0): Array of the log2FoldChange each cpg will be altered by in the simulated versus template samples.
            sample_size (integer, default: 100): Number of samples to simulate.
            
        Returns:
            2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
        """
        if vary_read_depth:
            coverage = np.tile(np.random.normal(read_depth,read_depth_sd,sample_size).astype(int),(params.shape[0],1))
        else:
            coverage=np.empty((params.shape[0],sample_size),dtype=int)
            coverage.fill(read_depth)
        coverage=np.clip(coverage, a_min=1,a_max=None)

        mu_wlink = np.matmul(
                X,
                params[:,:X.shape[1]].T
            )
        phi_wlink = np.matmul(
                X_star,
                params[:,-1*X_star.shape[1]:].T
            )
        if link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif link == "logit":
            mu = expit(mu_wlink)
            phi = expit(phi_wlink)
            
        if any(adjust_factors != 0):
            mu=mu+adjust_factors
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        else:
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        try:
            meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 
        except:
            raise ValueError("Parameters are not compatible with the beta binomial model. Likely adjust_factor is too high or something went wrong during fitting.")
        return meth, coverage

    @staticmethod
    def sim_internal(params,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100,link="arcsin"):
        """
        Simulate new samples based on existing samples, for this region.
    
        Parameters:
            params (numpy array): Array of paramether estimates (columns) for each cpg in the region (rows). 
            X (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the simulated samples for the 
            mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            X_star (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
            with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
            read_depth (integer, default: 30): Desired average read depth of simulated samples.
            vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
            read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
            adjust_factors (1D numpy array, default: 0): Array of the percent methylation each cpg will be altered by in the simulated versus template samples.
            sample_size (integer, default: 100): Number of samples to simulate.
            link (string, default: "arcsin"): Link function to use, either 'arcsin' or 'logit'.
            
        Returns:
            2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
        """
        if vary_read_depth: # Create array of sample read depths, varying read depth if specified to
            read_depth = np.random.normal(read_depth,read_depth_sd,sample_size).astype(int)
            coverage = np.tile(read_depth,(params.shape[0],1))    
        else:
            coverage=np.empty((params.shape[0],sample_size),dtype=int)
            coverage.fill(read_depth)
        coverage=np.clip(coverage, a_min=1,a_max=None)
        
        mu_wlink = np.matmul(
                X,
                params[:,:X.shape[1]].T
            )
        phi_wlink = np.matmul(
                X_star,
                params[:,-1*X_star.shape[1]:].T
            )
        if link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif link == "logit":
            mu = expit(mu_wlink)
            phi = expit(phi_wlink)
            
        if any(adjust_factors != 0):
            mu=mu+adjust_factors
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        else:
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi

        # Correct any out of bounds parameters
        a[a <= 0] = 1
        b[b <= 0] = 1
        
        # try:
        meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 
        # except:
        #     raise ValueError("Parameters are not compatible with the beta binomial model. Likely adjust_factor is too high or something went wrong during fitting.")
        return meth, coverage
        """
        Gets results for one contrast and corrects for multiple testing.
    
        Parameters:
            res (pandas dataframe): Results from running bbseq.
            contrast (string): Name of covariate you want the results from
            alpha (float): padj threshold below which results are considered statistically significant.
            padj_method (string): Method to adjust p values for multiple testing.
            
        Returns:
            numpy array: Summarised Array.
        """
        if res == "region":
            res = self.region_res
        else:
            res = self.cpg_res
        res = res[res.index == contrast]
        sig = multipletests(res["p"][~res["p"].isna()], alpha, padj_method)
        with pd.option_context('mode.chained_assignment', None):
            res["padj"] = 1.0
            res.loc[~res["p"].isna(),"padj"] = sig[1]
            res["sig"] = False
            res.loc[~res["p"].isna(),"sig"] = sig[0]
            
        return res

    def region_plot(self,region: int,contrast: str|list="",beta_vals: np.ndarray=np.array([]),show_codistrib_regions=True,dmrs=None,smooth=False):
        """
        Plot methylation level within a region.
    
        Parameters:
            region (float, integer, or string): Name of this region.
            contrast (optional: string): Name of column in self.X (covs) or list denoting group membership. Groups methylation levels over a region will be plotted as seperate lines.
            beta_vals (optional: np.array): Optional array of beta values to plot if wishing to plot samples not in object (e.g. simulated samples). Must have equal number of cpgs (rows)
            to the meth and coverage data input to the object (object.ncpgs).
            show_codistrib_regions (bool, default: True): Whether to show codistributed regions as shaded areas on the plot.
        """

        if dmrs is not None:
            assert isinstance(dmrs, pd.DataFrame), "dmrs must be a pandas dataframe"
            assert all(col in dmrs.columns for col in ["start", "end"]), "dmrs must contain columns named 'start' and 'end'"
            assert is_integer_dtype(dmrs['start']), "The 'start' column is not of integer dtype."
            assert is_integer_dtype(dmrs['end']), "The 'end' column is not of integer dtype."
            assert all(dmrs["start"] < dmrs["end"]), "dmrs column 'start' must be less than 'end'"
            assert all(dmrs["start"] >= 0), "dmrs column 'start' must be greater than or equal to 0"
            show_codistrib_regions = False
        if show_codistrib_regions:
            assert len(self.codistrib_regions) > 0, "Run bbseq with dmrs=True to find codistributed regions before setting show_codistrib_regions=True"
        if smooth:
            params = self.fits[np.where(self.individual_regions == region)[0][0]]
            mu_wlink = self.X @ params[:,:self.X.shape[1]].T
            if self.link == "arcsin":
                mu = np.sin(mu_wlink) ** 2
            elif self.link == "logit":
                mu = expit(mu_wlink)
            region_plot = pd.DataFrame(mu.T)
        else:
            if beta_vals.shape[0] == 0:
            #Compute beta values if required
                if isinstance(self.beta_vals,list):
                    self.beta_vals = self.meth / self.coverage
                region_plot = pd.DataFrame(self.beta_vals[self.target_regions == region])
            else:
                assert beta_vals.shape[0] == self.ncpgs, "beta_vals must contain the same number of cpgs (rows) as are present in the original object data"
                region_plot = pd.DataFrame(beta_vals[self.target_regions == region])
        
        if len(contrast) > 0:
            if isinstance(contrast, str):
                contrast = self.X[contrast].to_list()
            assert len(contrast) == region_plot.shape[1], "length of contrast must be equal to the number of samples"
            region_plot.columns=contrast
            means = {group: region_plot.loc[:,region_plot.columns==group].mean(axis=1) for group in self.unique(contrast)}
        else:
            means = region_plot.mean(axis=1)
        if self.genomic_positions is not None:
            region_plot["CpG"] = self.genomic_positions[self.target_regions==region]
            xlab = "Chromosome Position (bp)"
        else:
            region_plot["CpG"] = range(0,region_plot.shape[0])
            xlab = "CpG Number in Region"

        if show_codistrib_regions:
            assert len(self.codistrib_regions) == self.ncpgs, "Run bbseq before simulating if codistrib_regions=True"
            unique_codistrib_regions = np.array(self.unique(self.codistrib_regions[self.target_regions == region]))
            region_colours=["lightgrey","#5bd7f0", "lightgreen","#f0e15b", "#f0995b", "#db6b6b", "#cd5bf0", "#34b1eb", "#9934eb"] + [f"#{random.randrange(0x1000000):06x}" for _ in range(max(0,len(unique_codistrib_regions)-9))]
            codistrib_regions = self.codistrib_regions[self.target_regions == region]
            cdict = {x: region_colours[i] for i,x in enumerate(unique_codistrib_regions[unique_codistrib_regions!="0"])}
            cdict["0"] = "white"
            region_plot["region"] = [cdict[x] for x in codistrib_regions]
            region_plot=pd.melt(region_plot, id_vars=["CpG","region"],value_name='Beta Value')
        elif dmrs is not None:
            dmrs_regions = np.hstack([dmrs.loc[(dmrs["start"] <= pos) & (dmrs["end"] >= pos), ["chr","start","end"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).values
                         if any((pos >= dmrs["start"]) & (pos <= dmrs["end"])) 
                         else "0" for pos in self.genomic_positions[self.target_regions == region]])
            unique_dmrs_regions = np.array(self.unique(dmrs_regions))
            region_colours=(["lightgrey","#5bd7f0", "lightgreen","#f0e15b", "#f0995b", "#db6b6b", "#cd5bf0", "#34b1eb", "#9934eb"] + 
                            [f"#{random.randrange(0x1000000):06x}" for _ in range(max(0,len(unique_dmrs_regions)-9))])
            cdict = {x: region_colours[i] for i,x in enumerate(unique_dmrs_regions[unique_dmrs_regions!="0"])}
            cdict["0"] = "white"
            region_plot["region"] = [cdict[x] for x in dmrs_regions]
            region_plot=pd.melt(region_plot, id_vars=["CpG","region"],value_name='Beta Value')
        else:
            region_plot=pd.melt(region_plot, id_vars=["CpG"],value_name='Beta Value')
        if len(contrast) == 0:
            ax = sns.lineplot(region_plot, x="CpG", y="Beta Value")
        else:
            ax = sns.lineplot(region_plot, x="CpG", y="Beta Value", hue="variable")
        if show_codistrib_regions or dmrs is not None:
            ranges = region_plot.groupby('region')['CpG'].agg(['min', 'max'])
            for i, row in ranges.iterrows():
                ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=i, alpha=0.3)
            region_plot.loc[region_plot["region"].isna(),"region"] = "#4a4a4d"
            if len(contrast) == 0:
                sns.scatterplot(region_plot, x="CpG", y=means, c=region_plot.drop_duplicates("CpG")["region"], ax=ax,edgecolor="black")
            else:
                for group in self.unique(contrast):
                    sns.scatterplot(region_plot[region_plot["variable"]==self.unique(contrast)[0]], x="CpG", y=means[group], 
                                    c=region_plot[region_plot["variable"]==self.unique(contrast)[0]].drop_duplicates("CpG")["region"], ax=ax,edgecolor="black")
        if show_codistrib_regions:
            ax.set_title(f"Codistributed Regions in Region {region}")
        elif dmrs is not None:
            ax.set_title(f"DMRs in Region {region}")
        else:
            if len(contrast) == 0:
                sns.scatterplot(region_plot, x="CpG", y=means, ax=ax)
            else:
                for group in self.unique(contrast):
                    sns.scatterplot(region_plot[region_plot["variable"]==self.unique(contrast)[0]], x="CpG", y=means[group], ax=ax,edgecolor="black")
            ax.set_title(f"Region {region}")
        plt.xlabel(xlab)

    @staticmethod
    def unique(sequence):
        """
        Return unique elements of a list in the same order as their first occurance in the original list.
    
        Parameters:
            sequence (list): List
            
        Returns:
            List: Deduplicated list in the same order as items first occurance in the original list.
        """
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    @staticmethod
    @ray.remote
    def sum_regions(array):
        """
        Aggregates array values within predefined regions.
    
        Parameters:
            regions (numpy array): Regions each column of array belongs to (p-dimensional vector).
            array (numpy array): Array to be summarised.
            return_order (boolean, default: False): Whether to return the order of regions in the summarised array (can change from regions).
            
        Returns:
            numpy array: Summarised Array.
        """
        return np.nanmean(array, axis=1)

    def min_max(self,region,param_type="abd",type="both"):
        split = region.split("_")
        region = int(split[0])
        split = split[1].split("-")
        start = int(split[0])
        end = int(split[1])+1

        if param_type == "abd":
            param_type = 0
        else: 
            param_type = self.n_param_abd
            
        if self.link == "arcsin":
            params=np.sin(self.fits[np.sum(self.individual_regions < region)][start:end][:,param_type])**2
        else:
            params=expit(self.fits[np.sum(self.individual_regions < region)][start:end][:,param_type])
        if type=="both":
            return params.min(),params.max()
        elif type=="max":
            return params.max()
        
    def copy(self):
        """
        Create a deep copy of the pyMethObj instance.

        Returns:
            pyMethObj: A deep copy of the current instance.
        """
        return copy.deepcopy(self)