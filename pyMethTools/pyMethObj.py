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
        Initialize pyMethObj Class with methylation data and configuration parameters.

        Parameters:
            meth (2D numpy array): Count table of methylated reads at each CpG (rows) for each sample (columns).
            coverage (2D numpy array): Count table of total reads at each CpG (rows) for each sample (columns).
            target_regions (1D numpy array): Array specifying which target region each CpG belongs to.
                Must have same length as number of rows in meth/coverage.
            genomic_positions (1D numpy array or None, default: None): Genomic position of each CpG.
                Must have same length as number of rows in meth/coverage if provided.
            chr (1D numpy array or None, default: None): Chromosome identifiers for each CpG.
                If None, all positions are assigned to "chr".
            covs (pandas DataFrame or numpy array, optional): Design matrix for the mean parameter.
                Shape (number samples, number covariates) with one column per covariate, 
                including an intercept column (usually 1's named 'intercept').
                If not supplied, only an intercept column will be created.
            covs_disp (pandas DataFrame or numpy array, optional): Design matrix for the dispersion parameter.
                Shape (number samples, number covariates) with one column per covariate,
                including an intercept column (usually 1's named 'intercept').
                If not supplied, only an intercept column will be created.
            phi_init (float, default: 0.5): Initial value for the dispersion parameter.
            maxiter (int, default: 500): Maximum number of iterations for model fitting.
            maxfev (int, default: 500): Maximum number of function evaluations for model fitting.
            sample_weights (array-like or None, default: None): Optional weights for samples in the likelihood calculations.

        Raises:
            ValueError: If the model is overspecified (more parameters than samples).
            AssertionError: If input validation fails.

        Notes:
            Categorical covariates should be one-hot encoded before being passed to this function.
            The best way to provide covariates is to use Patsy dmatrix:
            # For an object fitted with design matrix for two treatments (control, treated) and two timepoints (T1, T2):
            #   covs columns will be: Intercept, treatment[T.treated], timepoint[T.T2],
            #   treatment[T.treated]:timepoint[T.T2]
            from patsy import dmatrix
            covs = dmatrix(
                "treatment + timepoint + treatment:timepoint",
                data=sample_traits, # Dataframe with sample traits (including treatment + timepoint columns)
                return_type='dataframe'
            )
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
        Fit beta binomial model to DNA methylation data, with optional parallel processing.
    
        Parameters:
            ncpu (int, default: 1): Number of CPU cores to use for parallel processing. 
                Values > 1 will use Ray for parallelization.
            start_params (list, default: []): Initial parameter values for optimization. 
                If empty, default initialization is used.
            maxiter (int, default: None): Maximum number of iterations for optimization.
                If None, uses the value specified during object initialization.
            maxfev (int, default: None): Maximum number of function evaluations for optimization.
                If None, uses the value specified during object initialization.
            link (str, default: "arcsin"): Link function for the mean parameter.
                Options: "arcsin" or "logit".
            fit_method (str, default: "gls"): Method used for fitting the model.
                Options include "gls" (generalized least squares) or "mle" (mmaximum likelihood estimation).
            chunksize (int, default: 1): Number of regions to process in each parallel task.
                Larger values may improve performance with many small regions.
            X (array-like, default: None): Alternative design matrix for the mean parameter.
                If None, uses the design matrix specified during object initialization.
            return_params (bool, default: False): If True, returns the fitted parameters 
                instead of storing them in the object.
            
        Returns:
            If return_params is True, returns a list of fitted parameter arrays.
            Otherwise, stores results in self.fits and self.se.
            
        Notes:
            - This method fits a beta-binomial model to each CpG, estimating parameters 
              that describe methylation probability and dispersion.
            - When using parallel processing (ncpu > 1), the Ray framework distributes 
              computation across available cores.
            - Larger chunksize values can improve performance by reducing communication 
              overhead, especially when processing many small regions.
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
        Process a chunk of target regions in parallel for beta-binomial model fitting.
        
        This remote function is designed to be executed via Ray for parallel processing. It takes
        a subset of regions and performs model fitting for all CpGs in those regions.
        
        Parameters:
            region_id (numpy array): Array of region assignments for the CpGs in this chunk.
            meth_id (ray.ObjectRef): Reference to methylation count data (shared via ray.put).
            coverage_id (ray.ObjectRef): Reference to coverage count data (shared via ray.put).
            X_id (ray.ObjectRef): Reference to design matrix for mean parameters (shared via ray.put).
            X_star_id (ray.ObjectRef): Reference to design matrix for dispersion parameters (shared via ray.put).
            fit_region (function): Function to fit the model to a region.
            fit_cpg (function): Function to fit the model to a single CpG.
            unique (function): Helper function to get unique elements while preserving order.
            maxiter (int, default: 150): Maximum number of iterations for optimization.
            maxfev (int, default: 150): Maximum number of function evaluations.
            start_params (list, default: []): Initial parameter values for optimization.
            param_names_abd (list, default: ["intercept"]): Names of mean parameters.
            param_names_disp (list, default: ["intercept"]): Names of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            sample_weights (array-like or None, default: None): Optional weights for samples.
            
        Returns:
            tuple: Contains two lists - the fitted parameters and standard errors for all CpGs
                  in the processed regions.
                  
        Notes:
            This function is designed to be called via ray.remote to enable parallel processing
            across multiple cores. The region_id array determines which CpGs are processed together.
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
        Fit beta-binomial models to all CpGs within a region.
        
        This function iterates through each CpG in the region and applies the fit_cpg function
        to estimate parameters for the beta-binomial model. It then combines the results into
        matrices of parameter estimates and standard errors.
        
        Parameters:
            meth_id (numpy array): Count table of methylated reads for each CpG in the region.
            coverage_id (numpy array): Count table of total reads for each CpG in the region.
            X_id (numpy array): Design matrix for the mean parameter.
            X_star_id (numpy array): Design matrix for the dispersion parameter.
            fit_cpg (function): Function to fit the beta-binomial model to a single CpG.
            maxiter (int, default: 150): Maximum number of iterations for optimization.
            maxfev (int, default: 150): Maximum number of function evaluations.
            start_params (list, default: []): Initial parameter values for optimization.
            param_names_abd (list, default: ["intercept"]): Names of mean parameters.
            param_names_disp (list, default: ["intercept"]): Names of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            sample_weights (array-like or None, default: None): Optional weights for samples.
            
        Returns:
            tuple: Contains two numpy arrays - the fitted parameters and standard errors for all CpGs
                  in the region. Each array has shape (n_cpgs, n_params).
                  
        Notes:
            This function processes all CpGs in a region sequentially. It is typically called
            by fit_betabinom_chunk during parallel processing or directly by fit_betabinom
            when processing regions without parallelization.
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
        Fit beta-binomial models to a specific region in parallel using Ray.
        
        This remote function is parallelized through Ray to enable concurrent 
        processing of different regions. It iterates through each CpG in the 
        specified region and applies the fit_cpg function to estimate parameters.
        
        Parameters:
            cpgs (array-like): Indices of CpGs belonging to this region.
            meth_id (ray.ObjectRef): Ray reference to the methylation count data.
            coverage_id (ray.ObjectRef): Ray reference to the coverage count data.
            X_id (ray.ObjectRef): Ray reference to the design matrix for mean parameters.
            X_star_id (ray.ObjectRef): Ray reference to the design matrix for dispersion parameters.
            fit_cpg (function): Function to fit the model to a single CpG.
            maxiter (int, default: 150): Maximum number of iterations for optimization.
            maxfev (int, default: 150): Maximum number of function evaluations.
            start_params (list, default: []): Initial parameter values for optimization.
            param_names_abd (list, default: ["intercept"]): Names of mean parameters.
            param_names_disp (list, default: ["intercept"]): Names of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            sample_weights (array-like or None, default: None): Optional weights for samples.
            
        Returns:
            tuple: Lists of fitted parameters and standard errors for all CpGs in the region.
            
        Notes:
            This function is designed to be called remotely via ray.remote to enable
            parallel processing. The function processes all CpGs belonging to a specific
            region (target region) and returns the combined results.
        """
        fits,se = zip(*[fit_cpg(meth_id[cpg],coverage_id[cpg],X_id,X_star_id,maxiter,maxfev,start_params,param_names_abd,param_names_disp,
                                       link,fit_method,sample_weights) 
                              for cpg in cpgs])
        
        return fits,se
    
    @staticmethod
    def fit_cpg_internal(meth,coverage,X,X_star,maxiter=150,maxfev=150,start_params=[],param_names_abd=["intercept"],param_names_disp=["intercept"],
                link="arcsin",fit_method="gls",sample_weights=None):
        """
        Fit a beta-binomial model to a single CpG site.
        
        This function handles the core model fitting for a single CpG, creating and
        optimizing a FitCpG object with the provided data. It's used internally by 
        both parallel and sequential fitting methods.
        
        Parameters:
            meth (array-like): Methylated read counts for one CpG across all samples.
            coverage (array-like): Total read counts for one CpG across all samples.
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            maxiter (int, default: 150): Maximum number of iterations for optimization.
            maxfev (int, default: 150): Maximum number of function evaluations.
            start_params (list, default: []): Initial parameter values for optimization.
            param_names_abd (list, default: ["intercept"]): Names of mean parameters.
            param_names_disp (list, default: ["intercept"]): Names of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            sample_weights (array-like or None, default: None): Optional weights for samples.
            
        Returns:
            tuple: Contains two arrays - the fitted parameters (x) and their standard errors (se_beta0).
            
        Notes:
            This is a utility function that wraps the FitCpG class to handle the actual model fitting.
            It's used by both the parallel and sequential implementations of beta-binomial fitting.
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
        Fit a beta-binomial model to a single CpG site using object instance data.
        
        This method is a non-parallelized version of the CpG fitting function that 
        operates directly on the object's data. It handles missing values and uses the 
        object's configuration for model fitting.
        
        Parameters:
            X (numpy array): Design matrix for the mean parameter.
            cpg (int): Index of the CpG site to fit in the object's data arrays.
            start_params (list, default: []): Initial parameter values for optimization.
            
        Returns:
            tuple: Contains two arrays - the fitted parameters (x) and their standard errors (se_beta0).
            
        Notes:
            This method filters out missing values (NaN) before fitting the model and
            uses the object's link function, fitting method, and other parameters stored
            in the instance attributes. It's primarily used by the non-parallel
            implementation of the beta-binomial fitting process.
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
        Fit beta-binomial models to all CpGs in a region using non-parallelized processing.
        
        This method processes all CpGs belonging to a specific region sequentially,
        applying the fit_cpg_local method to each one. It's used for fitting models
        when parallel processing is not enabled (ncpu=1).
        
        Parameters:
            X (numpy array): Design matrix for the mean parameter.
            cpgs (array-like): Indices of all CpGs in the dataset.
            region (int or str): Identifier for the region to process.
            start_params (list, default: []): Initial parameter values for optimization.
            
        Returns:
            tuple: Contains two numpy arrays - the fitted parameters and standard errors
                  for all CpGs in the region. Each array has shape (n_cpgs, n_params).
                  
        Notes:
            This method filters the CpGs to only include those that belong to the specified
            region before processing. It's used as an alternative to the ray-based parallel
            implementation when running on a single CPU.
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
        Fit a beta-binomial model to a single CpG site using Ray for parallel processing.
        
        This remote function handles the model fitting for a single CpG in a parallel 
        computing environment. It's designed to be executed within Ray's parallel 
        computing framework.
        
        Parameters:
            cpg (int): Index of the CpG site to fit.
            meth (numpy array): Methylation count data for all CpGs.
            coverage (numpy array): Coverage count data for all CpGs.
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            maxiter (int, default: 150): Maximum number of iterations for optimization.
            maxfev (int, default: 150): Maximum number of function evaluations.
            start_params (list, default: []): Initial parameter values for optimization.
            param_names_abd (list, default: ["intercept"]): Names of mean parameters.
            param_names_disp (list, default: ["intercept"]): Names of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            sample_weights (array-like or None, default: None): Optional weights for samples.
            
        Returns:
            tuple: Contains two arrays - the fitted parameters (x) and their standard errors (se_beta0).
            
        Notes:
            This function is decorated with @ray.remote to enable distributed processing.
            It creates a FitCpG object with the specific CpG's data and returns the 
            optimization results after fitting the beta-binomial model.
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
        """
        Smooth the fitted beta-binomial parameters across genomic positions.
        
        This method applies a spatial smoothing algorithm to the fitted parameters
        to reduce noise and improve parameter estimates by borrowing information
        from nearby CpGs. The smoothing is weighted based on genomic distance
        and parameter similarity.
        
        Parameters:
            lambda_factor (float, default: 10): Scaling factor that controls the
                strength of smoothing. Higher values result in more smoothing.
            param (str, default: "intercept"): Parameter name to use for calculating
                similarity weights in the smoothing process.
            ncpu (int, default: 1): Number of CPU cores to use for parallel processing.
                Values > 1 will use Ray for parallelization.
            chunksize (int, default: 1): Number of regions to process in each parallel task.
                Larger values may improve performance with many small regions.
                
        Returns:
            None: The smoothed parameters replace the original parameters in self.fits.
            
        Notes:
            - This method applies a custom kernel smoothing approach that weights observations 
              by both genomic proximity and parameter similarity.
            - The smoothing process respects region boundaries (doesn't smooth across regions).
            - Coverage depth is considered in the weighting to give less influence to
              low-coverage positions.
            - When ncpu > 1, processing is parallelized using Ray.
        """
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
        Process a chunk of regions for parallel smoothing of fitted parameters.
        
        This remote function is designed to be executed via Ray for parallel processing. 
        It takes a subset of regions and performs parameter smoothing for all CpGs in 
        those regions.
        
        Parameters:
            region_id (numpy array): Array of region assignments for the CpGs in this chunk.
            fits (list): Fitted parameters for CpGs in the target regions.
            n_param_abd (int): Number of abundance parameters in the model.
            genomic_positions (numpy array): Genomic positions of CpGs in the target regions.
            coverage (numpy array): Coverage data for CpGs in the target regions.
            smooth_region (function): Function to smooth parameters for a single region.
            unique (function): Helper function to get unique elements while preserving order.
            lambda_factor (float): Scaling factor controlling smoothing strength.
            link (str, default: "arcsin"): Link function used in the model.
            
        Returns:
            list: Smoothed parameter fits for all regions in this chunk.
            
        Notes:
            This function is designed to be called via ray.remote to enable parallel
            processing across multiple cores. It applies smoothing to each region
            in the chunk independently.
        """
        fits = [smooth_region(fits[reg_num],param,param_names,n_param_abd,genomic_positions[region_id==region],np.nanmean(coverage[region_id==region],axis=1),lambda_factor,link) 
                         for reg_num,region in enumerate(unique(region_id))]
        
        return fits
        
    @staticmethod
    def smooth_region_local(beta,param,param_names,n_par_abd,genomic_positions,cov,lambda_factor=10,link= "arcsin"):
        """
        Smooth fitted parameters for a region using a kernel-based approach.
        
        This function applies spatial smoothing to the parameter estimates within a region.
        Smoothing is weighted by genomic distance, parameter similarity, and coverage depth
        to improve parameter estimates by borrowing information from nearby CpGs.
        
        Parameters:
            beta (numpy array): Array of fitted parameters for CpGs in the region.
            param (str): Parameter name to use for calculating similarity weights.
            param_names (list): Names of all parameters in the model.
            n_par_abd (int): Number of abundance parameters in the model.
            genomic_positions (numpy array): Genomic positions of CpGs in the region.
            cov (numpy array): Mean coverage depth for each CpG in the region.
            lambda_factor (float, default: 10): Scaling factor controlling smoothing strength.
            link (str, default: "arcsin"): Link function used in the model.
            
        Returns:
            numpy array: Smoothed parameter estimates for the region.
            
        Notes:
            The smoothing algorithm uses three types of weights:
            1. Genomic weights: Based on genomic distance between CpGs
            2. Parameter weights: Based on similarity of the parameter values
            3. Coverage weights: Giving less influence to low-coverage positions
            
            The final weights are a product of these three weight components, and are
            used to compute a weighted average of the parameters across CpGs.
        """
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
        """
        Smooth fitted parameters for a region using Ray for parallel processing.
        
        This remote function applies spatial smoothing to parameter estimates within
        a region using a kernel-based approach. It's designed to be executed within
        Ray's parallel computing framework.
        
        Parameters:
            beta (numpy array): Array of fitted parameters for CpGs in the region.
            n_par_abd (int): Number of abundance parameters in the model.
            genomic_positions (numpy array): Genomic positions of CpGs in the region.
            cov (numpy array): Mean coverage depth for each CpG in the region.
            lambda_factor (float, default: 10): Scaling factor controlling smoothing strength.
            link (str, default: "arcsin"): Link function used in the model.
            
        Returns:
            numpy array: Smoothed parameter estimates for the region.
            
        Notes:
            This function is decorated with @ray.remote to enable distributed processing.
            It uses the same kernel-based smoothing approach as smooth_region_local, but
            only considers the intercept parameter for similarity weighting (first column
            of the parameter matrix).
            
            The smoothing process is applied twice consecutively to further reduce noise.
        """
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
        Identify co-distributed regions of CpGs with similar methylation patterns.
        
        This method identifies regions of contiguous CpGs that show similar methylation
        patterns and statistical properties, suggesting they are co-regulated. It combines
        CpGs that have similar beta-binomial parameters by testing whether they can be
        modeled with a single set of parameters without significant loss of fit.
        
        Parameters:
            site_names (numpy array, default: empty array): Optional array containing names for each CpG.
                Must have the same length as the number of CpGs if provided.
                If not provided, CpGs will be indexed by their position.
            dmrs (bool, default: True): Whether to identify differentially methylated regions.
            tol (int, default: 3): Tolerance factor for merging similar CpGs into regions.
                Higher values result in more aggressive merging.
            min_cpgs (int, default: 3): Minimum number of CpGs required to form a region.
            ncpu (int, default: 1): Number of CPU cores to use for parallel processing.
                Values > 1 will use Ray for parallelization.
            maxiter (int, default: 500): Maximum number of iterations for optimization when
                testing joint models for regions.
            maxfev (int, default: 500): Maximum number of function evaluations for optimization.
            chunksize (int, default: 1): Number of regions to process in each parallel task.
                Larger values may improve performance with many small regions.
                
        Returns:
            None: Results are stored in self.codistrib_regions as region labels for each CpG.
            
        Notes:
            - The method compares the fit of modeling CpGs separately versus jointly
              using the Bayesian Information Criterion (BIC).
            - CpGs are merged into regions when their joint model doesn't significantly
              reduce the fit quality compared to individual models.
            - The 'tol' parameter controls the sensitivity: higher values make the algorithm
              more likely to merge CpGs into larger regions.
            - Regions must contain at least min_cpgs+1 CpGs to be considered a valid region.
            - When parallel processing is used (ncpu > 1), regions are distributed
              across workers for faster computation.
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
        Process a chunk of regions to identify co-distributed CpG regions in parallel.
        
        This remote function is designed to be executed via Ray for parallel processing.
        It takes a subset of target regions and identifies co-distributed CpG regions
        within each of them.
        
        Parameters:
            region_id (numpy array): Array of region assignments for the CpGs in this chunk.
            find_codistrib (function): Function to identify co-distributed regions.
            beta_binomial_log_likelihood (function): Function to compute log-likelihood.
            unique (function): Helper function to get unique elements while preserving order.
            fits (list): Fitted parameters for CpGs in the target regions.
            meth (numpy array): Methylation count data for CpGs in the target regions.
            coverage (numpy array): Coverage count data for CpGs in the target regions.
            X_id (numpy array): Design matrix for the mean parameter.
            X_star_id (numpy array): Design matrix for the dispersion parameter.
            tol (int, default: 3): Tolerance factor for merging similar CpGs.
            min_cpgs (int, default: 3): Minimum number of CpGs required to form a region.
            maxiter (int, default: 150): Maximum iterations for optimization.
            maxfev (int, default: 150): Maximum function evaluations for optimization.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            
        Returns:
            numpy array: Array of region labels for each CpG in the processed chunk.
            
        Notes:
            This function is designed to be called via ray.remote to enable parallel
            processing across multiple cores. It distributes regions to be processed
            in parallel and aggregates the results.
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
        Identify co-distributed regions of CpGs within a single target region.
        
        This function analyzes contiguous CpGs within a region to identify segments
        that have similar methylation patterns and can be modeled with a single set 
        of parameters. It compares the goodness of fit between modeling CpGs separately
        versus jointly using the Bayesian Information Criterion (BIC).
        
        Parameters:
            region (int, float, or str): Identifier for the region being analyzed.
            beta_binomial_log_likelihood (function): Function to compute log-likelihood.
            fits (numpy array): Array of fitted parameters for CpGs in this region.
            meth (numpy array): Methylation count data for CpGs in this region.
            coverage (numpy array): Coverage count data for CpGs in this region.
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            tol (int, default: 3): Tolerance factor for merging similar CpGs.
                Higher values result in more aggressive merging.
            min_cpgs (int, default: 3): Minimum number of CpGs required to form a region.
            maxiter (int, default: 500): Maximum iterations for optimization.
            maxfev (int, default: 500): Maximum function evaluations for optimization.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            
        Returns:
            numpy array: Array of region labels for each CpG in the analyzed region.
            
        Notes:
            - The function uses a greedy algorithm that extends regions by adding CpGs
              one at a time and tests whether the joint model is still adequate.
            - CpGs are grouped into a region when their joint BIC is better than the
              penalized BIC of modeling them separately (BIC_separate * tol).
            - Regions must contain at least min_cpgs+1 CpGs to be considered a valid region.
            - Regions are labeled with a format: "{region}_{start}-{end}"
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
        Identify co-distributed regions of CpGs within a single target region using Ray.
        
        This remote function is a parallelized version of find_codistrib_local designed
        to be executed via Ray's distributed computing framework. It analyzes contiguous
        CpGs within a region to identify segments that share similar methylation patterns.
        
        Parameters:
            region (int, float, or str): Identifier for the region being analyzed.
            beta_binomial_log_likelihood (function): Function to compute log-likelihood.
            fits (numpy array): Array of fitted parameters for CpGs in this region.
            meth (numpy array): Methylation count data for CpGs in this region.
            coverage (numpy array): Coverage count data for CpGs in this region.
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            tol (int, default: 3): Tolerance factor for merging similar CpGs.
                Higher values result in more aggressive merging.
            min_cpgs (int, default: 3): Minimum number of CpGs required to form a region.
            maxiter (int, default: 500): Maximum iterations for optimization.
            maxfev (int, default: 500): Maximum function evaluations for optimization.
            link (str, default: "arcsin"): Link function for the mean parameter.
            fit_method (str, default: "gls"): Method for fitting the model.
            
        Returns:
            numpy array: Array of region labels for each CpG in the analyzed region.
            
        Notes:
            This function is decorated with @ray.remote to enable distributed processing
            across multiple cores. The algorithm is identical to find_codistrib_local,
            but this version is designed for parallel execution within Ray's framework.
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
        Compute the log-likelihood for beta-binomial regression.
        
        This function calculates the log-likelihood of methylation count data under
        a beta-binomial model with specified parameters. It handles various link functions
        and includes numerical stability safeguards.
        
        Parameters:
            count (array-like): Count of methylated reads for each sample.
            total (array-like): Total number of reads (coverage) for each sample.
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            beta (numpy array): Parameter vector containing concatenated mean (beta_mu)
                and dispersion (beta_phi) parameters.
            n_param_abd (int): Number of mean parameters.
            n_param_disp (int): Number of dispersion parameters.
            link (str, default: "arcsin"): Link function for the mean parameter.
                Options: "arcsin" or "logit".
            max_param (float, default: 1e+10): Maximum allowed value for beta distribution
                parameters to prevent numerical issues.
                
        Returns:
            float: Log-likelihood value of the data under the specified model.
            
        Notes:
            - The function first removes any NaN values from the input data.
            - It transforms linear predictors to probability space using the specified link function.
            - Parameters are scaled to avoid numerical overflow in the beta-binomial calculation.
            - The log-likelihood is computed using scipy.stats.betabinom.logpmf in vectorized form.
            - A higher log-likelihood indicates a better fit of the model to the data.
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
    def find_subregions(self, coef, tol=0.05, max_dist=10000, start_group=0):
        """
        Segment CpGs into regions based on parameter similarity using numba-accelerated code.
        
        This method divides CpGs into regions where the specified coefficient has
        similar values, with boundaries created when values change by more than the
        tolerance threshold. It also handles isolated outliers by giving them their
        own regions without breaking the main segment.
        
        Parameters:
            coef (str): Name of the coefficient to use for segmentation (must be in
                self.param_names_abd).
            tol (float, default: 0.05): Tolerance threshold for what constitutes a 
                significant change in parameter values.
            max_dist (int, default: 10000): Maximum genomic distance between adjacent
                CpGs to consider them part of the same region.
            start_group (int, default: 0): Starting group number for labeling regions.
                
        Returns:
            numpy.ndarray: Array of integer labels assigning each CpG to a region.
                
        Notes:
            - This method is accelerated using numba's just-in-time compilation (@njit).
            - Segmentation is based on the following rules:
              * Normal segmentation occurs when a value differs from the baseline or
                previous value by more than the tolerance.
              * If a single value exceeds tolerance but the next value returns to baseline,
                the outlier gets its own group without breaking the main segment.
              * New segments are also started at chromosome boundaries or when the
                genomic distance exceeds max_dist.
            - Useful for identifying regions of similar methylation patterns that may
              correspond to functional elements.
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
        Simulate methylation data based on fitted beta-binomial models.
        
        This method generates simulated methylation count data based on the parameters
        estimated from real data, with options to modify methylation levels in specific
        regions. It can be used for power analysis, benchmarking, or exploring the
        effects of differential methylation.
        
        Parameters:
            covs (pandas DataFrame or None, default: None): Design matrix for the mean parameter
                in the simulated data. If None, all covariates except intercept are set to 0.
            covs_disp (pandas DataFrame or None, default: None): Design matrix for the dispersion
                parameter. If None, all covariates except intercept are set to 0.
            use_codistrib_regions (bool, default: True): Whether to modify methylation in whole
                co-distributed regions (True) or individual CpGs (False).
            read_depth (str, int, or numpy array, default: "from_data"): Read depth for simulated data.
                If "from_data", uses average depths from the original data.
            vary_read_depth (bool, default: True): Whether to add variation to read depths.
            read_depth_sd (str, int, float, or numpy array, default: "from_data"): Standard deviation
                of read depths. If "from_data", uses the observed SD from original data.
            adjust_factor (float or list, default: 0): Amount to adjust methylation probability in
                differential regions. If list, specifies [increase, decrease] amounts.
            diff_regions_up (list or numpy array, default: []): Names of regions to increase methylation.
            diff_regions_down (list or numpy array, default: []): Names of regions to decrease methylation.
            n_diff_regions (int or list, default: 0): Number of random regions to differentially methylate.
                If list, specifies [n_up, n_down].
            prop_pos (float, default: 0.5): Proportion of differentially methylated regions to
                increase (vs. decrease) when using n_diff_regions.
            sample_size (int, default: 100): Number of samples to simulate.
            ncpu (int, default: 1): Number of CPU cores for parallel processing.
            chunksize (int, default: 1): Number of regions to process in each parallel task.
                
        Returns:
            tuple: If adjust_factor is 0, returns (simulated methylation counts, simulated coverage).
                  Otherwise, returns (simulated methylation counts, simulated coverage,
                  adjustment factors, list of modified regions).
                
        Notes:
            - Requires the model to be fitted first (self.fits must be populated).
            - Simulates data using the same beta-binomial model used for fitting.
            - When use_codistrib_regions=True and n_diff_regions>0, automatically selects
              suitable regions for modification based on their current methylation levels
              and dispersion parameters.
            - Parallelization via Ray is used when ncpu > 1.
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
            assert len(self.codistrib_regions) == self.ncpgs, "Run bbseq with dmrs=True to find codistributed regions before setting show_codistrib_regions=True"
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
        Simulate data for a chunk of regions in parallel using Ray.
        
        This remote function simulates methylation data for a set of target regions
        in parallel. It's designed to be executed within Ray's distributed computing
        framework for efficient parallel processing of large datasets.
        
        Parameters:
            region_id (array-like): Identifiers for the regions in this chunk.
            fits (list): Fitted model parameters for each region in the chunk.
            covs (pandas DataFrame): Design matrix for the mean parameter.
            covs_disp (pandas DataFrame): Design matrix for the dispersion parameter.
            sim_local (function): Function to simulate data for a single region.
            read_depth (array-like): Read depths for each region in the chunk.
            vary_read_depth (bool): Whether to add variation to read depths.
            read_depth_sd (array-like): Standard deviations for read depths.
            adjust_factors (array-like): Methylation adjustment factors for each CpG in each region.
            sample_size (int, default: 100): Number of samples to simulate.
            link (str, default: "arcsin"): Link function used in the model.
            
        Returns:
            tuple: Contains two numpy arrays - simulated methylation counts and
                  simulated coverage counts for all regions in the chunk.
                  
        Notes:
            This function is decorated with @ray.remote to enable distributed processing
            across multiple cores. It applies the simulation function to each region
            in the chunk and combines the results.
        """
        sim_meth, sim_coverage = zip(*[sim_local(fits[region_num],covs,covs_disp,
                                                          read_depth[region_num],vary_read_depth,read_depth_sd[region_num],
                                                     adjust_factors[region_num],sample_size,link) for region_num in range(len(region_id))])
        
        sim_meth=np.vstack(sim_meth)
        sim_coverage=np.vstack(sim_coverage)
        
        return sim_meth, sim_coverage

    def sim_local(self,region_num,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100):
        """
        Simulate methylation data for a single region using object's parameters.
        
        This method generates simulated methylation data for a specific region using
        the fitted parameters stored in the object. It uses the beta-binomial model
        to generate random counts that follow the statistical properties of the
        original data.
        
        Parameters:
            region_num (int): Index of the region to simulate data for.
            X (pandas DataFrame): Design matrix for the mean parameter.
            X_star (pandas DataFrame): Design matrix for the dispersion parameter.
            read_depth (int, default: 30): Mean read depth for the simulated data.
            vary_read_depth (bool, default: True): Whether to vary read depths across samples.
            read_depth_sd (float, default: 2): Standard deviation for read depth variation.
            adjust_factors (array-like, default: 0): Adjustment factors for methylation
                probability for each CpG in the region.
            sample_size (int, default: 100): Number of samples to simulate.
            
        Returns:
            tuple: Contains two numpy arrays - simulated methylation counts and
                  simulated coverage counts for the region.
                  
        Notes:
            - Uses the object's fitted parameters for the specified region.
            - Applies the object's link function to transform linear predictors.
            - If adjust_factors contains non-zero values, methylation probabilities
              are adjusted accordingly, allowing simulation of differential methylation.
            - Read depths are randomly generated from a normal distribution when
              vary_read_depth is True.
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
        Simulate methylation data for a region using Ray for parallel processing.
        
        This remote function generates simulated methylation data for a region with
        the provided parameters. It's designed to be executed within Ray's distributed
        computing framework for parallel processing across multiple cores.
        
        Parameters:
            params (numpy array): Array of fitted parameters (rows: CpGs, columns: parameters).
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            read_depth (int, default: 30): Mean read depth for the simulated data.
            vary_read_depth (bool, default: True): Whether to vary read depths across samples.
            read_depth_sd (float, default: 2): Standard deviation for read depth variation.
            adjust_factors (array-like, default: 0): Adjustment factors for methylation
                probability for each CpG in the region.
            sample_size (int, default: 100): Number of samples to simulate.
            link (str, default: "arcsin"): Link function for transforming linear predictors.
            
        Returns:
            tuple: Contains two numpy arrays - simulated methylation counts and
                  simulated coverage counts for the region.
                  
        Notes:
            This function is decorated with @ray.remote to enable distributed processing.
            It performs the same simulation process as sim_local and sim_internal, but
            is designed to be called remotely via Ray's distributed computing framework.
            
            The beta-binomial distribution is used to generate counts that reflect both
            the expected methylation probability and the expected biological variation.
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
        Simulate methylation data for internal processing using shared parameters.
        
        This utility function generates simulated methylation data for regions during
        parallel processing. It shares the same core functionality as sim and sim_local 
        but is designed for direct use within the parallel processing framework.
        
        Parameters:
            params (numpy array): Array of fitted parameters (rows: CpGs, columns: parameters).
            X (numpy array): Design matrix for the mean parameter.
            X_star (numpy array): Design matrix for the dispersion parameter.
            read_depth (int, default: 30): Mean read depth for the simulated data.
            vary_read_depth (bool, default: True): Whether to vary read depths across samples.
            read_depth_sd (float, default: 2): Standard deviation for read depth variation.
            adjust_factors (array-like, default: 0): Adjustment factors for methylation
                probability for each CpG in the region.
            sample_size (int, default: 100): Number of samples to simulate.
            link (str, default: "arcsin"): Link function for transforming linear predictors.
            
        Returns:
            tuple: Contains two numpy arrays - simulated methylation counts and
                  simulated coverage counts for the region.
                  
        Notes:
            - This function handles the core simulation logic that's shared across
              the various simulation methods (local, parallel, and chunked).
            - Simulated data follows a beta-binomial distribution with parameters
              derived from the fitted model parameters.
            - Methylation probabilities can be adjusted using adjust_factors to
              simulate differential methylation.
            - Coverage depth can be fixed or randomly varied based on the vary_read_depth
              parameter.
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
        Plot methylation levels across a genomic region with optional grouping and highlighting.
        
        This method creates a visualization of methylation patterns within a specific region.
        It can display raw beta values or model-smoothed estimates, highlight co-distributed
        regions or DMRs, and separate samples into groups based on a contrast variable.
        
        Parameters:
            region (int, float, or str): Identifier for the region to plot.
            contrast (str or list, default: ""): Variable to group samples by. Can be a column
                name from the design matrix (self.X) or a list of group assignments.
                When provided, separate lines are drawn for each group.
            beta_vals (numpy array, default: empty array): Optional array of beta values to plot.
                Used for plotting simulated or external methylation data. Must have the same
                number of CpGs as the original data if provided.
            show_codistrib_regions (bool, default: True): Whether to highlight co-distributed
                regions identified by find_codistrib_regions as colored background spans.
            dmrs (pandas DataFrame or None, default: None): Optional DataFrame of DMRs with
                columns 'chr', 'start', and 'end' to highlight on the plot instead of
                co-distributed regions.
            smooth (bool, default: False): Whether to plot model-smoothed methylation estimates
                instead of raw beta values.
                
        Returns:
            None: Displays a plot using matplotlib/seaborn.
            
        Notes:
            - If genomic positions are available, the x-axis represents genomic coordinates;
              otherwise, it shows CpG indices.
            - Co-distributed regions or DMRs are shown as color-coded background spans.
            - Points are plotted at each CpG position and connected with lines.
            - For grouped data, separate lines are shown for each group with a legend.
            - The method automatically handles missing values and ensures appropriate
              data ranges for visualization.
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
        Return unique elements of a list while preserving their original order.
        
        This utility function removes duplicate elements from a sequence while
        maintaining the order in which elements first appeared. Unlike set() which
        doesn't preserve order, this function ensures the order of first appearance
        is maintained.
        
        Parameters:
            sequence (list or array-like): The input sequence to deduplicate.
            
        Returns:
            list: A list of unique elements in their original order of first appearance.
            
        Notes:
            - This is implemented using a set to track seen elements for O(1) lookups.
            - The function processes the sequence in one pass, with O(n) time complexity.
            - Order preservation can be important for certain operations where the
              ordering of elements carries meaning (like region identifiers).
            
        Example:
            >>> unique([3, 1, 2, 1, 5, 3])
            [3, 1, 2, 5]
        """
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    @staticmethod
    @ray.remote
    def sum_regions(array):
        """
        Aggregate values across columns within an array using Ray for parallel processing.
        
        This remote function computes column-wise means for a given array while
        handling missing values. It's designed for parallel aggregation of data
        in distributed computing environments using Ray.
        
        Parameters:
            array (numpy array): Array to be summarized across columns.
            
        Returns:
            numpy array: Column means of the input array, ignoring NaN values.
            
        Notes:
            - This function is decorated with @ray.remote to enable distributed processing
              across multiple cores.
            - NaN values are handled gracefully using numpy's nanmean function.
            - Typically used to aggregate methylation metrics (like beta values)
              across samples within regions.
            - This function is most efficiently used when processing large arrays
              that benefit from parallelization.
        """
        return np.nanmean(array, axis=1)

    def min_max(self,region,param_type="abd",type="both"):
        """
        Calculate minimum and/or maximum parameter values for a co-distributed region.
        
        This utility method extracts and transforms parameter estimates for a specified
        co-distributed region and returns their minimum and/or maximum values. It's
        primarily used to determine the range of methylation probabilities within
        a region for simulation and visualization purposes.
        
        Parameters:
            region (str): Identifier for the co-distributed region, formatted as
                "{region_id}_{start}-{end}".
            param_type (str, default: "abd"): Type of parameter to extract:
                - "abd": Abundance/methylation parameter (typically intercept)
                - Any other value: Dispersion parameter
            type (str, default: "both"): Which value(s) to return:
                - "both": Return (min, max) as a tuple
                - "max": Return only the maximum value
                
        Returns:
            tuple or float: Depending on the 'type' parameter:
                - If type="both": (minimum value, maximum value)
                - If type="max": maximum value only
                
        Notes:
            - This method parses the region identifier to extract the base region ID,
              start position, and end position.
            - Parameters are transformed from link space to probability space using
              the appropriate link function (arcsin or logit).
            - Primarily used when selecting regions for simulation of differential
              methylation to ensure valid probabilities after adjustments.
        """
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
        Create a copy of the pyMethObj instance.

        Returns:
            pyMethObj: A copy of the current instance.
        """
        return copy.deepcopy(self)