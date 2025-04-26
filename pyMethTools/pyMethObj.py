import numpy as np
from numba import njit
import pandas as pd
from pandas.api.types import is_integer_dtype
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom,norm,chi2
from scipy.special import gammaln,binom,beta,logit,expit,digamma,polygamma
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.families.links import Link
from itertools import chain
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
from hmmlearn import hmm
from pyMethTools.FitCpG import FitCpG

class pyMethObj():
    """
    A Python class containing functions for analysis of targetted DNA methylation data.
    These include:
        Fitting beta binomial models to cpgs, inferring parameters for the mean abundance, dispersion, and any other given covariates: fit_betabinom().
        Smoothing fitted parameters: smooth().
        Differential methylation analysis by assessing the statistical significance of covariate parameters, while finding contigous regions of codistributed 
        cpgs with similar parameters, and testing if these are differentially methylated: find_codistrib_regions().
        Getting regression results for any contrast: get_contrast().
        Simulating new data based on fitted data: sim_multiple_cpgs().

    All methods allow parallel processing to speed up computation (set ncpu parameter > 1).
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

    def fit_betabinom(self,ncpu: int=1,start_params=[],maxiter=None,maxfev=None,link="arcsin",fit_method="gls",chunksize=1,X=None,
                      return_params=False):
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

        if X is None:
            X=self.X
    
        if ncpu > 1: #Use ray for parallel processing
            ray.init(num_cpus=ncpu)    
            X_id=ray.put(X)
            X_star_id=ray.put(self.X_star)
            if chunksize>1:
                fits,se = zip(*ray.get([self.fit_betabinom_chunk.remote(
                    self.target_regions[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.meth[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    self.coverage[np.isin(self.target_regions,self.individual_regions[chunk:chunk+chunksize])],
                    X_id,X_star_id,self.fit_region_internal,self.fit_cpg_internal,self.unique,
                    maxiter,maxfev,start_params,
                    self.param_names_abd,self.param_names_disp,self.link,
                    self.fit_method, self.sample_weights) 
                              for chunk in range(0,len(set(self.individual_regions)),chunksize)]))
                
                fits = list(chain.from_iterable(fits))
                se = list(chain.from_iterable(se))
             
            else:
                fits,se = zip(*ray.get([self.fit_region.remote(self.region_cpg_indices[self.target_regions==region],
                                                            self.meth[self.target_regions==region],self.coverage[self.target_regions==region],
                                                            X_id,X_star_id,self.fit_cpg_internal,maxiter,maxfev,start_params,self.param_names_abd,
                                                            self.param_names_disp,self.link,self.fit_method,self.sample_weights) 
                                  for region in self.individual_regions]))

            ray.shutdown()
            
        else:
            fits,se = zip(*[self.fit_region_local(X,cpgs,region,start_params) for region in self.individual_regions])

        if return_params:
            return fits
        else:
            self.fits = fits
            self.se = se


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
        cc = FitCpG(
                    total=coverage,
                    count=meth,
                    X=X,
                    X_star=X_star,
                    link=link,
                    fit_method=fit_method,
                    sample_weights=sample_weights
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev,start_params=start_params)
        return e_m.x,e_m.se_beta0
        
    def fit_cpg_local(self,X,cpg,start_params=[]):
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
        cc = FitCpG(
                    total=self.coverage[cpg][~np.isnan(self.meth[cpg])],
                    count=self.meth[cpg][~np.isnan(self.meth[cpg])],
                    X=X[~np.isnan(self.meth[cpg])],
                    X_star=self.X_star[~np.isnan(self.meth[cpg])],
                    link=self.link,
                    fit_method=self.fit_method,
                    sample_weights=self.sample_weights
                )
        e_m = cc.fit(maxiter=self.maxiter,maxfev=self.maxfev,start_params=start_params)
        return e_m.x,e_m.se_beta0

    def fit_region_local(self,X,cpgs,region,start_params=[]):
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

        fits,se = zip(*[self.fit_cpg_local(X,cpg,start_params) for cpg in cpgs[self.target_regions==region]])
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
        cc = FitCpG(
                    total=coverage[cpg],
                    count=meth[cpg],
                    X=X,
                    X_star=X_star,
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
    
    def likelihood_ratio_test(
            self,
            coef=None,
            contrast=None,
            padjust_method='fdr_bh',
            find_dmrs='binary_search',
            prop_sig=0.5,
            fdr_thresh=0.1,
            max_gap=10000,
            max_gap_cpgs=3,
            min_cpgs=3,
            n_states=3,
            state_labels=None,
            ncpu=1):
        """
        Compute LRT for beta-binomial regression fits via scipy.stats.betabinom.logpmf.

        Parameters
        ----------
        coef : str or list of str, optional
            Name or list of names of parameters to test. Mutually exclusive with `contrast`.
        contrast : array-like or dict, optional
            Contrast specification. Can be:
            - 1D array-like of length p (one contrast)
            - 2D array-like of shape (k, p) (k contrasts)
            - dict mapping param names to weights for a single contrast
            Mutually exclusive with `coef`.
        padjust_method : str
            p-value adjustment method (passed to statsmodels.stats.multitest.multipletests).
        find_dmrs : {'binary_search', 'HMM', None}, default 'binary_search'
            Method to identify differentially methylated regions (DMRs). Options are:
            - 'binary_search': Use binary search to find significant regions.
            - 'HMM': Use a Hidden Markov Model to identify regions.
            - None: Do not identify DMRs.
        prop_sig : float, default 0.5
            Proportion of significant CpGs required to define a DMR.
        fdr_thresh : float, default 0.1
            FDR threshold for identifying significant regions.
        max_gap : int, default 10000
            Maximum genomic distance (in base pairs) between CpGs to consider them part of the same region.
        max_gap_cpgs : int, default 3
            Maximum number of non-significant CpGs allowed between significant CpGs in a region.
        min_cpgs : int, default 3
            Minimum number of CpGs required to define a DMR.
        n_states : int, default 3
            Number of states for the Hidden Markov Model (if `find_dmrs='HMM'`).
        state_labels : list of str, optional
            Labels for the states in the Hidden Markov Model.
        ncpu : int, default 1
            Number of CPUs to use for parallel computation.

        Returns
        -------
        DataFrame
            Columns ['llf_full','llf_reduced','D','df','pval'], one row per CpG (column of params_*).
        """
        # Back-transform to probabilities
        params = np.vstack(self.fits)
        beta_mu = params[:,:self.n_param_abd]
        mu_wlink = self.X @ beta_mu.T
        prob_full = ((np.sin(mu_wlink) + 1) / 2) 

        # Fit reduced model
        if coef is not None:
            coef_indices = np.where(np.in1d(self.param_names, coef))[0]
        elif contrast is not None:
            coef_indices = np.where(np.in1d(self.param_names, list(contrast.keys())))[0]

        reduced_X = np.delete(self.X, coef_indices, axis=1)
        params_red = np.vstack(self.fit_betabinom(X=reduced_X,return_params=True,ncpu=ncpu))
        beta_mu_red = params_red[:,:reduced_X.shape[1]]
        mu_wlink_red = reduced_X @ beta_mu_red.T
        prob_red = ((np.sin(mu_wlink_red) + 1) / 2) 

        # Method-of-moments to alpha, beta: total = 1/phi - 1
        phi_inv = (1/params[:,self.n_param_abd]) - 1
        a_full = prob_full * phi_inv
        b_full = (1 - prob_full) * phi_inv
        phi_inv_red = 1/params_red[:,self.n_param_abd-(params.shape[1]-params_red.shape[1])] - 1
        a_red = prob_red * phi_inv_red
        b_red = (1 - prob_red) * phi_inv_red

        # Log-likelihood per CpG (sum over samples)
        ll_full = betabinom.logpmf(self.meth, self.coverage, a_full.T, b_full.T).sum(axis=1)
        ll_red = betabinom.logpmf(self.meth, self.coverage, a_red.T, b_red.T).sum(axis=1)

        # LRT statistic
        D = -2 * (ll_red - ll_full)
        df = beta_mu.shape[1] - beta_mu_red.shape[1]
        pval = chi2.sf(D, df)

        cpg_res = pd.DataFrame({
            'chr': self.chr,
            'pos': self.genomic_positions,
            'llf_full': ll_full,
            'llf_reduced': ll_red,
            'D': D,
            'df': df,
            'pval': pval,
            'fdr': multipletests(pval, method=padjust_method)[1],
        })

        if find_dmrs == 'binary_search':
            dmr_res = self.find_significant_regions(cpg_res,prop_sig=prop_sig,fdr_thresh=fdr_thresh,max_gap=max_gap,
                                                    max_gap_cpgs=max_gap_cpgs)
            return cpg_res, dmr_res
        
        elif find_dmrs == 'HMM':
            dmr_res = self.find_significant_regions_HMM(cpg_res, n_states=n_states, min_cpgs=min_cpgs, fdr_thresh=fdr_thresh, 
                                                        prop_sig_thresh=prop_sig, state_labels=state_labels, 
                                                        hmm_plots=False, hmm_internals=False)
            return cpg_res, dmr_res
        
        else:
            return cpg_res
    
    def wald_test(
        self,
        coef=None,
        contrast=None,
        padjust_method='fdr_bh',
        n_permute=0,
        find_dmrs='binary_search',
        prop_sig=0.5,
        fdr_thresh=0.1,
        max_gap=10000,
        max_gap_cpgs=3,
        min_cpgs=3,
        n_states=3,
        state_labels=None,
        ncpu=1
    ):
        """
        Perform Wald tests for individual coefficients or user-specified contrasts.

        Parameters
        ----------
        coef : str or list of str, optional
            Name or list of names of parameters to test. Mutually exclusive with `contrast`.
        contrast : array-like or dict, optional
            Contrast specification. Can be:
            - 1D array-like of length p (one contrast)
            - 2D array-like of shape (k, p) (k contrasts)
            - dict mapping param names to weights for a single contrast
            Mutually exclusive with `coef`.
        padjust_method : str
            p-value adjustment method (passed to statsmodels.stats.multitest.multipletests).
        n_permute : int, default 0
            Number of permutations for empirical p-value calculation. If 0, no permutation testing is performed.
        find_dmrs : {'binary_search', 'HMM', None}, default 'binary_search'
            Method to identify differentially methylated regions (DMRs). Options are:
            - 'binary_search': Use binary search to find significant regions.
            - 'HMM': Use a Hidden Markov Model to identify regions.
            - None: Do not identify DMRs.
        prop_sig : float, default 0.5
            Proportion of significant CpGs required to define a DMR.
        fdr_thresh : float, default 0.1
            FDR threshold for identifying significant regions.
        max_gap : int, default 10000
            Maximum genomic distance (in base pairs) between CpGs to consider them part of the same region.
        max_gap_cpgs : int, default 3
            Maximum number of non-significant CpGs allowed between significant CpGs in a region.
        min_cpgs : int, default 3
            Minimum number of CpGs required to define a DMR.
        n_states : int, default 3
            Number of states for the Hidden Markov Model (if `find_dmrs='HMM'`).
        state_labels : list of str, optional
            Labels for the states in the Hidden Markov Model.
        ncpu : int, default 1
            Number of CPUs to use for parallel computation.

        Returns
        -------
        DataFrame
            Results with columns ['contrast', 'stat', 'pval', 'fdr', 'chr', 'pos'].

        Examples
        --------
        # For an object fitted with design matrix for two treatments (control, treated) and two timepoints (T1, T2):
        #   X columns will be: Intercept, treatment[T.treated], timepoint[T.T2],
        #   treatment[T.treated]:timepoint[T.T2]
        from patsy import dmatrix
        X = dmatrix(
            "treatment + timepoint + treatment:timepoint",
            data=sample_traits,
            return_type='dataframe'
        )

        # 1. Test the main effect of treatment (treated vs. control):
        #    This tests the coefficient 'treatment[T.treated]'.
        res_treat = model.wald_test(coef='treatment[T.treated]')

        # 2. Test the main effect of timepoint (T2 vs. T1):
        res_time = model.wald_test(coef='timepoint[T.T2]')

        # 3. Test the interaction contrast: difference of treatment effect at T2 vs T1.
        #    Contrast: (treatment:timepoint at T2) - (no interaction at T1).
        res_int = model.wald_test(
            contrast={
                'treatment[T.treated]:timepoint[T.T2]': 1
            }
        )

        # 4. Compare treated vs control specifically at T2 (joint of main + interaction):
        #    Contrast vector = treatment + treatment:timepoint@T2
        res_trt_at_T2 = model.wald_test(
            contrast={
                'treatment[T.treated]': 1,
                'treatment[T.treated]:timepoint[T.T2]': 1
            }
        )"""
        # Validate inputs: exactly one of coef or contrast
        if (coef is None) == (contrast is None):
            raise ValueError("Specify exactly one of `coef` or `contrast`, but not both.")

        # Build contrast matrix C and labels
        p = self.X.shape[1]
        param_names = list(self.param_names_abd)

        if coef is not None:
            # single or multiple coefficient tests: identity contrasts
            if isinstance(coef, str):
                coef = [coef]
            idx = [param_names.index(c) for c in coef]
            C = np.eye(p)[idx]
            labels = coef
        else:
            # user-specified contrast
            if isinstance(contrast, dict):
                weights = np.zeros(p)
                for name, w in contrast.items():
                    weights[param_names.index(name)] = w
                C = weights.reshape(1, -1)
                labels = "+".join(contrast.keys())
            else:
                arr = np.asarray(contrast)
                if arr.ndim == 1:
                    C = arr.reshape(1, -1)
                    labels = "+".join(contrast.keys())
                elif arr.ndim == 2:
                    C = arr
                    labels = [f'contrast_{i}' for i in range(arr.shape[0])]
                else:
                    raise ValueError("`contrast` must be 1D or 2D array-like or dict.")

        # extract betas and ses for all parameters
        betas = np.vstack(self.fits)[:, :self.n_param_abd]
        ses = np.vstack(self.se)[:, :self.n_param_abd]

        results = []
        for row, label in zip(C, labels):
            effect = (betas * row).sum(axis=1)
            var_effect = (ses**2 * row**2).sum(axis=1)
            stat = effect / np.sqrt(var_effect)
            pval = 2 * norm.sf(np.abs(stat))
            fdr = multipletests(pval, method=padjust_method)[1]

            df = pd.DataFrame({
                'contrast': label,
                'stat': stat,
                'pval': pval,
                'fdr': fdr,
                'chr': self.chr,
                'pos': self.genomic_positions
            })
            results.append(df)

        res = pd.concat(results, axis=0)

        if n_permute > 1:
            # Permute the labels of the specified column
            if coef is not None:
                perm_stats = self.permute_and_refit(coef, N=n_permute, ncpu=ncpu)
            elif contrast is not None:
                perm_stats = self.permute_and_refit(contrast=contrast, N=n_permute, ncpu=ncpu)
            # Calculate the empirical p-value (two-sided)
            emp_pval = np.mean(perm_stats["stats"].values >= stat[:, np.newaxis], axis=1)

            res["pval"] = emp_pval
            res["fdr"] = multipletests(emp_pval, method=padjust_method)[1]
            
        if find_dmrs == 'binary_search':
            dmr_res = self.find_significant_regions(res,prop_sig=prop_sig,fdr_thresh=fdr_thresh,max_gap=max_gap,
                                                    max_gap_cpgs=max_gap_cpgs)
            if (n_permute > 1) and (not dmr_res.empty):
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
                                            'pval': perm_stats["pval"].values[:,perm],
                                            'fdr': perm_stats["fdr"].values[:,perm],
                                            'emp_pval': emp_pval,
                                            'emp_fdr': multipletests(emp_pval, method=padjust_method)[1]
                                        })
                    permuted_dmrs.append(self.find_significant_regions(perm_res,prop_sig=prop_sig,
                                                                       fdr_thresh=fdr_thresh,max_gap=max_gap,
                                                                       min_cpgs=min_cpgs,
                                                                       max_gap_cpgs=max_gap_cpgs))

                # Concatenate the permuted DMRs, if none found add one with 0 cpgs
                permuted_prop_sig_cpgs = [perm.prop_sig_cpgs.values if not perm.empty else np.array([0]) for perm in permuted_dmrs]
                permuted_prop_sig_cpgs = np.hstack(permuted_prop_sig_cpgs)
                # Calculate the empirical p-value - how many times did we see such a region in the permutations
                emp_pval = [np.mean(permuted_prop_sig_cpgs >= region_stats) for region_stats in dmr_res["prop_sig_cpgs"]]
                dmr_res["pval"] = emp_pval
                dmr_res["fdr"] = multipletests(emp_pval, method=padjust_method)[1]

            return res, dmr_res
        
        elif find_dmrs == 'HMM':
            dmr_res = self.find_significant_regions_HMM(res, n_states=n_states, min_cpgs=min_cpgs, fdr_thresh=fdr_thresh, 
                                                        prop_sig_thresh=prop_sig, state_labels=state_labels, 
                                                        hmm_plots=False, hmm_internals=False)
            return res, dmr_res

        else:
            return res


    @staticmethod
    def find_significant_regions(df, prop_sig=0.5, fdr_thresh=0.1, maxthresh=0.2,
                                max_gap=1000, min_cpgs=3, max_gap_cpgs=2):
        """
        Find regions of adjacent CpGs where:
        - The gap between successive CpGs is less than max_gap.
        - The region is extended as far as possible until a gap violation, an FDR above maxthresh, 
            or more than max_gap_cpgs adjacent non-significant CpGs would be introduced.
        - The region must end on a significant CpG (FDR <= fdr_thresh).
        - Among candidate regions starting from a given significant CpG, the candidate with the greatest 
            number of significant CpGs is chosen; ties are broken by selecting the one with the fewest 
            non-significant CpGs.
        - The region must contain at least min_cpgs CpGs.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns 'chr', 'pos', and 'fdr'.
            max_gap (int): Maximum allowed gap between adjacent CpGs.
            prop_sig (float): Minimum overall proportion of CpGs that must be significant (FDR <= fdr_thresh).
            fdr_thresh (float): FDR threshold for significance.
            maxthresh (float): FDR threshold above which a region immediately ends.
            min_cpgs (int): Minimum number of CpGs required in a region.
            max_gap_cpgs (int): Maximum number of adjacent non-significant CpGs allowed in a region.
        
        Returns:
            pd.DataFrame: DataFrame with columns 'chr', 'start', 'end', 'num_cpgs',
                        'num_sig_cpgs', and 'prop_sig_cpgs' for each region.
        """
        significant_regions = []
        
        # Sort the DataFrame by chromosome and position.
        df = df.sort_values(by=['chr', 'pos']).reset_index(drop=True)
        
        # Process each chromosome separately.
        for chr_name, group in df.groupby('chr'):
            # Reset index for 0-based indexing.
            chr_df = group.reset_index(drop=True)
            positions = chr_df['pos'].to_numpy()
            n = len(chr_df)
            used_cpgs = set()  # to avoid overlaps
            
            i = 0
            while i < n:
                # Start only if the current CpG is significant and not already used.
                if chr_df['fdr'].iloc[i] > fdr_thresh or i in used_cpgs:
                    i += 1
                    continue
                
                # Extend the region as far as possible from the starting point i.
                j_max = i
                while j_max + 1 < n:
                    # Check the gap constraint.
                    if positions[j_max + 1] - positions[j_max] > max_gap:
                        break
                    # End if the next CpG's FDR is above maxthresh.
                    if chr_df['fdr'].iloc[j_max + 1] > maxthresh:
                        break
                    
                    # Check if including the next CpG would create a block of > max_gap_cpgs adjacent non-sig CpGs.
                    candidate = chr_df['fdr'].iloc[i:j_max+2].to_numpy()  # region from i to j_max+1
                    current_count = 0
                    max_adj_non_sig = 0
                    for val in candidate:
                        if val > fdr_thresh:
                            current_count += 1
                            max_adj_non_sig = max(max_adj_non_sig, current_count)
                        else:
                            current_count = 0
                    if max_adj_non_sig > max_gap_cpgs:
                        break
                    
                    j_max += 1
                
                # Now, consider all candidate endpoints from minimal region size (i + min_cpgs - 1) to j_max.
                # For each candidate, the region must end on a significant CpG.
                candidate_regions = []
                start_index = i
                for candidate_end in range(max(start_index + min_cpgs - 1, start_index), j_max + 1):
                    # Skip if the candidate endpoint is not a significant CpG.
                    if chr_df['fdr'].iloc[candidate_end] > fdr_thresh:
                        continue
                    
                    subregion = chr_df.iloc[start_index:candidate_end + 1]
                    num_cpgs = len(subregion)
                    num_sig = (subregion['fdr'] <= fdr_thresh).sum()
                    prop = num_sig / num_cpgs if num_cpgs > 0 else 0
                    
                    # Recompute maximum adjacent non-sig count in the candidate region.
                    candidate_vals = subregion['fdr'].to_numpy()
                    current_count = 0
                    max_adj_non_sig = 0
                    for val in candidate_vals:
                        if val > fdr_thresh:
                            current_count += 1
                            max_adj_non_sig = max(max_adj_non_sig, current_count)
                        else:
                            current_count = 0

                    # Accept candidate only if it meets the overall proportion and does not violate the adjacent rule.
                    if prop >= prop_sig and max_adj_non_sig <= max_gap_cpgs:
                        non_sig_count = num_cpgs - num_sig
                        # Record candidate as a tuple:
                        # (endpoint index, number of sig CpGs, number of non-sig CpGs, proportion, candidate region DataFrame)
                        candidate_regions.append((candidate_end, num_sig, non_sig_count, prop, subregion))
                
                # If any candidate region meets the criteria, select the one with the most significant CpGs.
                # Ties are broken by selecting the candidate with fewer non-significant CpGs;
                # a further tie-breaker is the candidate_end (lowest index).
                if candidate_regions:
                    candidate_regions.sort(key=lambda x: (-x[1], x[2], x[0]))
                    best_candidate_end, best_num_sig, best_non_sig, best_prop, best_region = candidate_regions[0]
                    num_cpgs = len(best_region)
                    final_prop = best_num_sig / num_cpgs if num_cpgs > 0 else 0
                    significant_regions.append({
                        'chr': chr_name,
                        'start': best_region['pos'].iloc[0],
                        'end': best_region['pos'].iloc[-1],
                        'num_cpgs': num_cpgs,
                        'num_sig_cpgs': best_num_sig,
                        'prop_sig_cpgs': final_prop
                    })
                    used_cpgs.update(range(i, best_candidate_end + 1))
                    i = best_candidate_end + 1
                else:
                    # If no candidate region meets the criteria, advance beyond the extended region.
                    i = j_max + 1

        return pd.DataFrame(significant_regions)


    
    @staticmethod
    def find_significant_regions_HMM(cpg_res, n_states=3, min_cpgs=5, fdr_thresh=0.05, prop_sig_thresh=0.5, 
                                max_gap=5000, state_labels=None, hmm_plots=False, hmm_internals=False, ncpu=4):
        """
        Identify candidate DMR regions using an HMM with multiple states and multivariate features.
        Utilizes Ray for parallel processing of chromosomes only, not segments.
        
        Parameters
        ----------
        cpg_res : pd.DataFrame
            Must contain the following columns:
            - 'chr': Chromosome identifier
            - 'pos': Genomic position
            - 'pval': CpG pvalue
            - 'fdr': CpG fdr
            - 'stat': CpG Test Statistic
        n_states : int, default 3
            Number of HMM states (0: Background, 1: Hypermethylated, 2: Hypomethylated)
        min_cpgs : int, default 5
            Minimum number of consecutive CpGs to report a region
        fdr_thresh : float, default 0.05
            FDR threshold for significance
        prop_sig_thresh : float, default 0.5
            Minimum proportion of significant CpGs required
        max_gap : int, default 5000
            Maximum allowed gap between adjacent CpGs in a region
        state_labels : dict or None
            Optional mapping from state index to state name
        hmm_plots : bool, default False
            Whether to generate plots for HMM state decoding
        hmm_internals : bool, default False
            Whether to print internal HMM parameters
        ncpu : int, default 4
            Number of CPU cores to use for parallel processing
            
        Returns
        -------
        regions_df : pd.DataFrame
            Candidate regions with chromosome, position, and statistical information
        """
        if state_labels is None:
            state_labels = {0: 'Background', 1: 'Hypermethylated', 2: 'Hypomethylated'}
        
        candidate_regions = []
        cpg_res['-log10pval'] = -np.log10(cpg_res['pval'])
        
        # Initialize Ray for parallel processing if not already initialized
        ray_initialized = ray.is_initialized()
        if not ray_initialized:
            ray.init(num_cpus=ncpu)
        
        @ray.remote
        def process_chromosome(chrom_group, cpg_res_full):
            """Process a single chromosome's CpGs to identify DMR regions"""
            group = chrom_group.sort_values('pos').reset_index(drop=True)
            chrom = group['chr'].iloc[0]
            
            # Create observations array with both score1 and score2
            obs = group[['-log10pval', 'stat']].values  # shape: (n_samples, 2)
            
            # Initialize a Gaussian HMM with three states and diagonal covariance
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", 
                                n_iter=100, random_state=42, init_params="c")
            
            model.transmat_ = np.array([
                [0.9999, 0.00005, 0.00005],  # From background
                [0.005, 0.99, 0.005],        # From hyper
                [0.005, 0.005, 0.99]         # From hypo
            ])
            
            # Initialize means
            model.means_ = np.array([
                [0, 0.0],     # State 0: Background 
                [3.0, 3.0],   # State 1: Hypermethylated 
                [3.0, -3.0]   # State 2: Hypomethylated
            ])
            
            # Initialize diagonal covariances
            model.covars_ = np.tile(np.array([1.0, 1.0]), (n_states, 1))
            
            # Fit the HMM to the observations
            model.fit(obs)
            
            if hmm_internals:
                print("hmm state means:\n", model.means_)
                print("hmm state transition probabilities:\n", model.transmat_)
            
            # Decode the most likely state sequence
            state_seq = model.predict(obs)
            group['state'] = state_seq
            
            # Optional plot for inspection
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
            
            # Process segments sequentially (no parallel processing within chromosome)
            local_regions = []
            current_region = None
            
            for i, row in group.iterrows():
                st = row['state']
                
                if st != 0:  # non-background state
                    if current_region is None:
                        current_region = {
                            'chr': chrom,
                            'start_idx': i,
                            'end_idx': i,
                            'state': st
                        }
                    else:
                        # Check if gap exceeds max_gap
                        if i > 0 and (row['pos'] - group.loc[i-1, 'pos'] > max_gap):
                            # End current region if long enough
                            if (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                                start_pos = group.loc[current_region['start_idx'], 'pos']
                                end_pos = group.loc[current_region['end_idx'], 'pos']
                                
                                # Count significant CpGs in this region
                                num_sig = (cpg_res_full.loc[(cpg_res_full.chr == chrom) &
                                                        (cpg_res_full.pos >= start_pos) &
                                                        (cpg_res_full.pos <= end_pos), 'fdr'] <= fdr_thresh).sum()
                                
                                local_regions.append({
                                    'chr': chrom,
                                    'start': start_pos,
                                    'end': end_pos,
                                    'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                                    'num_sig': num_sig,
                                    'state': state_labels[current_region['state']]
                                })
                            # Start a new region
                            current_region = {
                                'chr': chrom,
                                'start_idx': i,
                                'end_idx': i,
                                'state': st
                            }
                        elif st == current_region['state']:
                            current_region['end_idx'] = i
                        else:
                            # End current region if long enough and start a new one
                            if (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                                start_pos = group.loc[current_region['start_idx'], 'pos']
                                end_pos = group.loc[current_region['end_idx'], 'pos']
                                
                                # Count significant CpGs in this region
                                num_sig = (cpg_res_full.loc[(cpg_res_full.chr == chrom) &
                                                        (cpg_res_full.pos >= start_pos) &
                                                        (cpg_res_full.pos <= end_pos), 'fdr'] <= fdr_thresh).sum()
                                
                                local_regions.append({
                                    'chr': chrom,
                                    'start': start_pos,
                                    'end': end_pos,
                                    'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                                    'num_sig': num_sig,
                                    'state': state_labels[current_region['state']]
                                })
                            current_region = {
                                'chr': chrom,
                                'start_idx': i,
                                'end_idx': i,
                                'state': st
                            }
                else:  # Background state
                    if current_region is not None:
                        if (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                            start_pos = group.loc[current_region['start_idx'], 'pos']
                            end_pos = group.loc[current_region['end_idx'], 'pos']
                            
                            # Count significant CpGs in this region
                            num_sig = (cpg_res_full.loc[(cpg_res_full.chr == chrom) &
                                                    (cpg_res_full.pos >= start_pos) &
                                                    (cpg_res_full.pos <= end_pos), 'fdr'] <= fdr_thresh).sum()
                            
                            local_regions.append({
                                'chr': chrom,
                                'start': start_pos,
                                'end': end_pos,
                                'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                                'num_sig': num_sig,
                                'state': state_labels[current_region['state']]
                            })
                        current_region = None
            
            # Check if a region remains at the end
            if current_region is not None and (current_region['end_idx'] - current_region['start_idx'] + 1) >= min_cpgs:
                start_pos = group.loc[current_region['start_idx'], 'pos']
                end_pos = group.loc[current_region['end_idx'], 'pos']
                
                # Count significant CpGs in this region
                num_sig = (cpg_res_full.loc[(cpg_res_full.chr == chrom) &
                                        (cpg_res_full.pos >= start_pos) &
                                        (cpg_res_full.pos <= end_pos), 'fdr'] <= fdr_thresh).sum()
                
                local_regions.append({
                    'chr': chrom,
                    'start': start_pos,
                    'end': end_pos,
                    'num_cpgs': current_region['end_idx'] - current_region['start_idx'] + 1,
                    'num_sig': num_sig,
                    'state': state_labels[current_region['state']]
                })
            
            return local_regions
        
        # Process each chromosome in parallel
        chrom_tasks = []
        for chrom, group in cpg_res.groupby('chr'):
            chrom_tasks.append(process_chromosome.remote(group, cpg_res))
        
        # Collect results
        chrom_results = ray.get(chrom_tasks)
        
        # Combine all regions
        for chrom_regions in chrom_results:
            candidate_regions.extend(chrom_regions)
        
        # Clean up Ray if we initialized it
        if not ray_initialized:
            ray.shutdown()
        
        # Create DataFrame from candidate regions and filter by significance threshold
        region_res = pd.DataFrame(candidate_regions)
        if not region_res.empty:
            region_res['prop_sig'] = region_res['num_sig'] / region_res['num_cpgs']
            region_res = region_res[region_res['prop_sig'] >= prop_sig_thresh]
        
        return region_res

    def permute_and_refit(self, coef=None, contrast=None, N=100, padjust_method='fdr_bh', ncpu=1):
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
        permuted_stats = []
        permuted_pval = []
        permuted_fdr = []

        for i in range(N):
            # Permute the labels of the specified column
            permuted_obj = self.copy()
            permuted_obj.X = np.random.permutation(permuted_obj.X)

            # Refit the model
            permuted_obj.fit_betabinom(ncpu=ncpu)

            # Compute Wald test statistics
            if coef is not None:
                wald_res = permuted_obj.wald_test(coef, padjust_method=padjust_method, n_permute=0, find_dmrs=False)
            elif contrast is not None:
                wald_res = permuted_obj.wald_test(contrast=contrast, padjust_method=padjust_method, n_permute=0, find_dmrs=False)
            permuted_stats.append(wald_res['stat'].values)
            permuted_pval.append(wald_res['pval'].values)
            permuted_fdr.append(wald_res['fdr'].values)

        # Combine results into a DataFrame
        permuted_stats_df = pd.DataFrame(np.vstack(permuted_stats).T, columns=[f'perm_{i}' for i in range(N)])
        permuted_pval_df = pd.DataFrame(np.vstack(permuted_pval).T, columns=[f'perm_{i}' for i in range(N)])
        permuted_fdr_df = pd.DataFrame(np.vstack(permuted_fdr).T, columns=[f'perm_{i}' for i in range(N)])

        permuted_stats = {"stats": permuted_stats_df, "pval": permuted_pval_df, "fdr": permuted_fdr_df}
        return permuted_stats

    def find_codistrib_regions(self,site_names:np.ndarray=np.array([]),dmrs: bool=True,tol: int=3,min_cpgs: int=3,ncpu: int=1,maxiter=500,maxfev=500,chunksize=1):
        """
        Perform differential methylation analysis using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
        mean and association with covariates are similar.
        
        Parameters:
            site_names (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying cpg name. If not provided names each cpg by its index.
            dmrs (boolean, default: True): Whether to identify and test differentially methylated regions of contigous cpgs whose mean and association with covariates are similar.
            tol (integer, default: 3): Tolerance for merging of similarly distributed cpgs into regions (higher increases merging).
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
                    self.X,self.X_star,tol,min_cpgs,maxiter,maxfev,self.link,self.fit_method)
                    for chunk in range(0,len(set(self.individual_regions)),chunksize)])
            else:
                codistrib_regions = ray.get([self.find_codistrib.remote(region,self.beta_binomial_log_likelihood,self.fits[fit],
                                                                        self.meth[self.target_regions==region],
                                                                        self.coverage[self.target_regions==region],
                                                                        self.X,self.X_star,tol,min_cpgs,
                                                                        maxiter,maxfev,self.link,self.fit_method) 
                                                    for fit,region in enumerate(self.individual_regions)])
            ray.shutdown()
            
            self.codistrib_regions = np.hstack(codistrib_regions)
        
        else:
            
            codistrib_regions = [self.find_codistrib_local(region,self.beta_binomial_log_likelihood,self.fits[fit],self.meth[self.target_regions==region],
                                                            self.coverage[self.target_regions==region],self.X,self.X_star,tol,
                                                            min_cpgs,maxiter,maxfev,self.link,self.fit_method) 
                                                            for fit,region in enumerate(self.individual_regions)]
                    
            self.codistrib_regions = np.hstack(codistrib_regions)

    @staticmethod
    @ray.remote
    def find_codistrib_chunk(region_id,find_codistrib,beta_binomial_log_likelihood,unique,fits,meth,coverage,X_id,X_star_id,tol=3,
                             min_cpgs=3,maxiter=150,maxfev=150,link="arcsin",fit_method="gls"):
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
                                                 X_id,X_star_id,min_cpgs,maxiter,maxfev,link,fit_method) 
                                                   for fit,region in enumerate(unique(region_id))]
        
        codistrib_regions = np.hstack(codistrib_regions)
        
        return codistrib_regions


    @staticmethod
    def find_codistrib_local(region,beta_binomial_log_likelihood,fits,meth,coverage,X,X_star,tol=3,min_cpgs=3,maxiter=500,maxfev=500,
                             link="arcsin",fit_method="gls"):
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
        bad_cpgs=0  
        X_c=X
        X_star_c=X_star
        fits = fits
        n_params_abd=X.shape[1]
        n_params_disp=X_star.shape[1]
        n_params=n_params_abd+n_params_disp
        ll_c = beta_binomial_log_likelihood(meth[start],coverage[start],X,X_star,fits[start],n_params_abd,n_params_disp,link=link)
        
        while end < len(fits)+1:
            
            if start+min_cpgs < len(fits)+1: #Can we form a codistrib region
                X_c=np.vstack([X_c,X])
                X_star_c=np.vstack([X_star_c,X_star])
                ll_s = ll_c + beta_binomial_log_likelihood(meth[end-1],coverage[end-1],X,X_star,fits[end-1],
                                                           n_params_abd,n_params_disp,link=link)
                
                # Find joint region parameters
                fits_c = FitCpG(
                    total=coverage[start:end].flatten(),
                    count=meth[start:end].flatten(),
                    X=X_c,
                    X_star=X_star_c,
                    link=link,
                    fit_method=fit_method
                ).fit(maxiter=maxiter,maxfev=maxfev).x

                # Get log likelihood of joint region parameters
                ll_c = beta_binomial_log_likelihood(meth[start:end].flatten(),coverage[start:end].flatten(),X_c,X_star_c,
                                                    fits_c,n_params_abd,n_params_disp,link=link) 

                # Compare joint vs seperate parameters
                bic_c = -2 * ll_c + (n_params) * np.log(n_samples) 
                bic_s = -2 * ll_s + (n_params*tol) * np.log(n_samples) 
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
    def find_codistrib(region,beta_binomial_log_likelihood,fits,meth,coverage,X,X_star,tol=3,min_cpgs=3,maxiter=500,
                       maxfev=500,link="arcsin",fit_method="gls"):
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
        X_c=X
        X_star_c=X_star
        fits = fits
        n_params_abd=X.shape[1]
        n_params_disp=X_star.shape[1]
        ll_c = beta_binomial_log_likelihood(meth[start],coverage[start],X,X_star,fits[start],n_params_abd,n_params_disp,link=link)
        
        while end < len(fits)+1:
            
            if start+min_cpgs < len(fits)+1: #Can we form a codistrib region
                X_c=np.vstack([X_c,X])
                X_star_c=np.vstack([X_star_c,X_star])
                ll_s = ll_c + beta_binomial_log_likelihood(meth[end-1],coverage[end-1],X,X_star,fits[end-1],
                                                           n_params_abd,n_params_disp,link=link)
                
                # Find joint region parameters
                fits_c = FitCpG(
                    total=coverage[start:end].flatten(),
                    count=meth[start:end].flatten(),
                    X=X_c,
                    X_star=X_star_c,
                    link=link,
                    fit_method=fit_method
                ).fit(maxiter=maxiter,maxfev=maxfev).x

                # Get log likelihood of joint region parameters
                ll_c = beta_binomial_log_likelihood(meth[start:end].flatten(),coverage[start:end].flatten(),X_c,X_star_c,
                                                    fits_c,n_params_abd,n_params_disp,link=link) 

                # Compare joint vs seperate parameters
                bic_c = -2 * ll_c + (n_params) * np.log(n_samples) 
                bic_s = -2 * ll_s + (n_params*tol) * np.log(n_samples) 
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
        X=X[~np.isnan(total)]
        X_star=X_star[~np.isnan(total)]
        total=total[~np.isnan(total)]
        count=count[~np.isnan(total)]

        # Reshape beta into segments for mu and phi
        beta_mu = beta[:n_param_abd]
        beta_phi = beta[-n_param_disp:]
        
        # Compute linear predictors
        mu_wlink = X @ beta_mu.T
        phi = X_star @ beta_phi.T
    
        # Transform to scale (0, 1)
        if link == "arcsin":
            mu = ((np.sin(mu_wlink) + 1) / 2) 
        elif link == "logit":
            mu = expit(mu_wlink)
    
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
        log_likelihood = np.sum(betabinom.logpmf(count, total, a.T, b.T))

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
                        diff_regions_up = np.random.choice(self.codistrib_regions_only[(self.codistrib_region_max_beta+adj_pos < 0.9) & (self.disp_intercept < 0.1)],n_pos,replace=False)
                    except:
                        raise ValueError("Less than n_diff_regions regions with low enough probability methylation to be raised by adjust_factor without going above 100% methylated. Likely adjust_factor or n_diff_regions is too high.")
                    try:
                        diff_regions_down = np.random.choice(self.codistrib_regions_only[(self.codistrib_region_min_beta-adj_neg > 0.1) & (self.disp_intercept < 0.1)
                                                             & ~np.isin(self.codistrib_regions_only,diff_regions_up)],n_neg,replace=False)
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
            ) for chunk in range(0,len(set(self.individual_regions)),chunksize)]))
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
        
        return sim_meth, sim_coverage

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
        phi = np.matmul(
                X_star,
                params[:,self.n_param_abd:].T
            )
        if self.link == "arcsin":
            mu = ((np.sin(mu_wlink) + 1) / 2) 
        elif self.link == "logit":
            mu = expit(mu_wlink)
            
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

        meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 

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
        if vary_read_depth: # Create array of sample read depths, varying read depth if specified to
            read_depth = np.random.normal(read_depth,read_depth_sd,sample_size).astype(int)
            coverage = np.tile(read_depth,(params.shape[0],1))    
        else:
            coverage=np.empty((params.shape[0],sample_size),dtype=int)
            coverage.fill(read_depth)
        coverage=np.clip(coverage, a_min=1,a_max=None) # Ensure at least 1 coverage
        
        mu_wlink = np.matmul(
                X,
                params[:,:X.shape[1]].T
            )
        phi = np.matmul(
                X_star,
                params[:,X.shape[1]:].T
            )
        if link == "arcsin":
            mu = ((np.sin(mu_wlink) + 1) / 2) 
        elif link == "logit":
            mu = expit(mu_wlink)
            
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

        meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 

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
        coverage=np.clip(coverage, a_min=1,a_max=None) # Ensure at least 1 coverage
        
        mu_wlink = np.matmul(
                X,
                params[:,:X.shape[1]].T
            )
        phi = np.matmul(
                X_star,
                params[:,X.shape[1]:].T
            )
        if link == "arcsin":
            mu = ((np.sin(mu_wlink) + 1) / 2) 
        elif link == "logit":
            mu = expit(mu_wlink)
            
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

        meth = betabinom.rvs(coverage,a.T,b.T,size=(params.shape[0],sample_size)) 

        return meth, coverage

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
                mu = ((np.sin(mu_wlink) + 1) / 2)
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