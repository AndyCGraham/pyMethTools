import numpy as np
import pandas as pd
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta,logit,expit,digamma,polygamma
from statsmodels.stats.multitest import multipletests
from itertools import chain
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random


class pyMethObj():
    """
    A Python class containing functions for analysis of targetted DNA methylation data.
    These include:
        Fitting beta binomial models to cpgs, inferring parameters for the mean abundance, dispersion, and any other given covariates: fit_betabinom().
        Differential methylation analysis by assessing the statistical significance of covariate parameters, while finding contigous regions of codistributed 
        cpgs with similar parameters, and testing if these are differentially methylated: bbseq().
        Getting regression results for any contrast: get_contrast().
        Simulating new data based on fitted data: sim_multiple_cpgs().
    
    Beta binomial fitting to CpGs is based on a method implemented by pycorncob (https://github.com/jgolob/pycorncob/tree/main), and concieved by 
    https://doi.org/10.1214/19-AOAS1283, with some adjustments for DNA methylation data.

    Init Parameters:
        meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
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
    
    def __init__(self, meth: np.ndarray, coverage: np.ndarray, target_regions: np.ndarray, covs=None, covs_disp=None, phi_init: float=0.5, maxiter: int=250, maxfev: int=400):
        """
        Intitiate pyMethObj Class

        Parameters:
            meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
            coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
            target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
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
        assert isinstance(maxiter,int), "maxiter must be positive integer"
        assert maxiter>0, "maxiter must be positive integer"
        assert isinstance(maxfev,int), "maxfev must be positive integer"
        assert maxfev>0, "maxfev must be positive integer"
        
        self.meth = meth
        self.coverage = coverage
        self.ncpgs = meth.shape[0]
        self.target_regions = target_regions
        self.individual_regions = self.unique(target_regions)
        counts = {}
        self.region_cpg_indices = [counts[x]-1 for x in target_regions if not counts.update({x: counts.get(x, 0) + 1})]

        if not isinstance(covs, pd.DataFrame):
            covs = pd.DataFrame(np.repeat(1,meth.shape[1]))
            covs.columns=['intercept']

        assert covs.shape[0] == meth.shape[1], "covs must have one row per sample (column in meth)"
            
        if not isinstance(covs_disp, pd.DataFrame):
            covs_disp = pd.DataFrame(np.repeat(1,meth.shape[1]))
            covs_disp.columns=['intercept']

        assert covs_disp.shape[0] == meth.shape[1], "covs_disp must have one row per sample (column in meth)"
        
        self.X = covs
        self.X_star = covs_disp
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
        self.phi_init = phi_init
        self.maxiter = maxiter
        self.maxfev = maxfev
        
        # Inits
        self.fits=[]
        self.cometh=[]
        self.codistrib_regions=[]
        self.codistrib_region_beta_vals=[]
        self.codistrib_region_mean_beta = []
        self.beta_vals = []
        self.region_cov = []
        self.region_cov_sd = []

    def fit_betabinom(self,chunksize: int=1,ncpu: int=1):
        """
        Fit beta binomial model to DNA methylation data.
    
        Parameters:
            chunksize (integer, default: 1): Number of regions to process at once if using parallel processing.
            ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
            
        Returns:
            numpy array: Array of optimisation results of length equal to the number of cpgs. 
        """

        assert isinstance(chunksize,int), "chunksize must be positive integer"
        assert chunksize>0, "chunksize must be positive integer"
        assert isinstance(ncpu,int), "ncpu must be positive integer"
        assert ncpu>0, "ncpu must be positive integer"
    
        if ncpu > 1: #Use ray for parallel processing
            ray.init(num_cpus=ncpu)    
            meth_id=ray.put(self.meth)
            coverage_id=ray.put(self.coverage)
            X_id=ray.put(self.X)
            X_star_id=ray.put(self.X_star)
            region_id=ray.put(self.target_regions)
            if chunksize>1:
                result = ray.get([self.fit_betabinom_chunk.remote(individual_regions[chunk:chunk+chunksize],
                                                         meth_id,coverage_id,region_id,self.maxiter,self.maxfev) 
                              for chunk in range(0,len(set(self.individual_regions))+1,chunksize)])
                self.fits = np.array(chain.from_iterable(result))
            else:
                self.fits = np.array(ray.get([self.fit_cpg.remote(cpg,meth_id,coverage_id,X_id,X_star_id,self.maxiter,self.maxfev) 
                              for cpg in range(self.ncpgs)]))
            ray.shutdown()
            
        else:
            self.fits = np.array([self.fit_cpg_local(self.meth[cpg],self.coverage[cpg],self.X,self.X_star,self.maxiter,self.maxfev) 
                              for cpg in range(self.ncpgs)])

    @staticmethod
    @ray.remote
    def fit_betabinom_chunk(chunk_regions,meth_id,coverage_id,region_id,maxiter=250,maxfev=250):
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
        meth_id = meth_id[regions in chunk_regions]
        coverage_id = coverage_id[regions in chunk_regions]
        chunk_result = [fit_cpg_local(meth_id[cpg],coverage_id[cpg],maxiter,maxfev) 
                        for cpg in range(meth_id.shape[0])]
        return chunk_result

    @staticmethod
    @ray.remote
    def fit_cpg(cpg,meth_id,coverage_id,X_id,X_star_id,maxiter=250,maxfev=400):
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
                    total=coverage_id[cpg],
                    count=meth_id[cpg],
                    X=X_id,
                    X_star=X_star_id
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev)
        return cc
        
    @staticmethod
    def fit_cpg_local(meth,coverage,X,X_star,maxiter=250,maxfev=400):
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
                    total=coverage,
                    count=meth,
                    X=X,
                    X_star=X_star
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev)
        return cc

    def sim_multiple_cpgs(self,covs=None,covs_disp=None,use_codistrib_regions: bool=True,read_depth: str|int="from_data",vary_read_depth=True,read_depth_sd: str|int|float="from_data",
                          adjust_factor: float|list=0, diff_regions_up: list|np.ndarray=[],diff_regions_down: list|np.ndarray=[],n_diff_regions: list|int=0,prop_pos: float=0.5,sample_size: int=100,ncpu: int=1):
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

        assert len(self.fits) == self.ncpgs, "Run fit before simulating"
        assert any([read_depth == "from_data", isinstance(read_depth,(list,int))]), "read_depth must be 'from_data' or a positive integer"
        if isinstance(read_depth,(list,int)):
            assert all(read_depth>0) , "read_depth must be a positive integer"
        assert any([read_depth_sd == "from_data", isinstance(read_depth_sd,(list,float,int))]), "read_depth_sd must be 'from_data' or a positive integer or float"
        if isinstance(read_depth,(list,int)):
            assert read_depth_sd>0, "read_depth_sd must be positive"
        if isinstance(adjust_factor,float):
            assert all([adjust_factor <= 1, adjust_factor>=0]), "adjust_factor must be a float or list of floats of length 2 between 0 and 1"
        if isinstance(adjust_factor,list):
            assert all([adjust_factor < [1,1], adjust_factor > [0,0]]), "adjust_factor must be a float or list of floats of length 2 between 0 and 1"
        assert isinstance(ncpu,int), "ncpu must be positive integer"
        assert ncpu>0, "ncpu must be positive integer"

        if isinstance(read_depth,str):
            if len(self.region_cov) == 0:
                self.region_cov = np.array([self.coverage[self.target_regions==region].mean().astype(int) for region in self.individual_regions])
            read_depth = self.region_cov
        elif isinstance(read_depth,int):
            read_depth = np.repeat(read_depth,len(self.individual_regions))

        if isinstance(read_depth_sd,str):
            if len(self.region_cov_sd) == 0:
                self.region_cov_sd = np.array([self.coverage[self.target_regions==region].std() for region in self.individual_regions])
            read_depth_sd = self.region_cov_sd
        elif isinstance(read_depth_sd,(int,float)):
            read_depth_sd = np.repeat(read_depth_sd,len(self.individual_regions))

        if not isinstance(covs, pd.DataFrame):
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
                    independent_regions = np.array([region for region in self.unique(self.codistrib_regions) if "_" in region])
                    if isinstance(self.codistrib_region_beta_vals,list):
                        self.is_codistrib_region = np.isin(self.codistrib_regions,independent_regions)
                        meth = self.sum_regions(self.codistrib_regions[self.is_codistrib_region],self.meth[self.is_codistrib_region]) 
                        coverage = self.sum_regions(self.codistrib_regions[self.is_codistrib_region],self.coverage[self.is_codistrib_region]) 
                        self.codistrib_region_beta_vals = meth / coverage
                    if isinstance(self.codistrib_region_mean_beta,list):
                        self.codistrib_region_mean_beta = self.codistrib_region_beta_vals.mean(axis=1)
                    try:
                        diff_regions_up = np.random.choice(independent_regions[(self.codistrib_region_mean_beta<=np.median(self.codistrib_region_mean_beta)) & (self.codistrib_region_mean_beta+adj_pos < 1)],n_pos)
                    except:
                        try:
                            diff_regions_up = np.random.choice(independent_regions[self.codistrib_region_mean_beta+adj_pos < 0.99],n_pos)
                        except:
                            raise ValueError("Less than n_diff_regions regions with low enough probability methylation to be raised by adjust_factor without going above 100% methylated. Likely adjust_factor or n_diff_regions is too high.")
                    try:
                        diff_regions_down = np.random.choice(independent_regions[(self.codistrib_region_mean_beta>=np.median(self.codistrib_region_mean_beta)) & ~np.isin(independent_regions,diff_regions_up)],n_neg)
                    except:
                        try:
                            diff_regions_down = np.random.choice(independent_regions[(self.codistrib_region_mean_beta-adj_neg > 0.01) & ~np.isin(independent_regions,diff_regions_up) & (self.codistrib_region_mean_beta-adj_neg > 0)],n_neg)
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
        
        if ncpu > 1: #Use ray for parallel processing
            ray.init(num_cpus=ncpu)    
            sim_meth, sim_coverage = zip(*ray.get([self.sim.remote(np.vstack([fit.theta for fit in self.fits[self.target_regions==region]]),covs,covs_disp,
                                                          read_depth[self.individual_regions==region],vary_read_depth,read_depth_sd[self.individual_regions==region],
                                                     adjust_factors[self.target_regions==region],sample_size) for region in self.individual_regions]))
            ray.shutdown()

        else:
            sim_meth, sim_coverage = zip(*[self.sim_local(np.vstack([fit.theta for fit in self.fits[self.target_regions==region]]),covs,covs_disp,
                                                          read_depth[self.individual_regions==region],vary_read_depth,read_depth_sd[self.individual_regions==region],
                                                     adjust_factors[self.target_regions==region],sample_size) for region in self.individual_regions])
        return np.vstack(sim_meth), np.vstack(sim_coverage), adjust_factors

    @staticmethod
    def sim_local(params,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100):
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
        mu = expit(mu_wlink)
        phi = expit(phi_wlink) 
            
        if any(adjust_factors != 0):
            mu=mu+adjust_factors
            mu=mu.clip(upper=1)
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
    @ray.remote
    def sim(params,X,X_star,read_depth=30,vary_read_depth=True,read_depth_sd=2,adjust_factors=0,sample_size=100):
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
        mu = expit(mu_wlink)
        phi = expit(phi_wlink) 
            
        if any(adjust_factors != 0):
            mu=mu+adjust_factors
            mu=mu.clip(upper=1)
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

    def bbseq(self,site_names:np.ndarray=np.array([]),dmrs: bool=True,min_cpgs: int=3,ncpu: int=1):
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
        
            if dmrs:

                fits_id=ray.put(self.fits)
                site_names_id=ray.put(site_names)
                region_id=ray.put(self.target_regions)
                cpg_res,region_res = zip(*ray.get([self.corncob_region.remote(region,region_id,fits_id,min_cpgs) for region in self.individual_regions]))
                ray.shutdown()
                self.cpg_res = pd.concat(cpg_res)
                self.region_res = pd.concat(region_res)
                tmp=self.region_res[~self.region_res["site"].duplicated()]
                self.codistrib_regions = np.array([tmp[tmp["region"]==str(region)]["site"].iloc[next((i for i, sublist in enumerate(tmp[tmp["region"]==str(region)]["range"]) if pos in sublist ), -1)]
                 if any(int(pos) in reg for reg in tmp[tmp["region"]==str(region)]["range"]) else site_names[cpg] for pos,cpg,region in zip(self.region_cpg_indices,range(self.ncpgs), self.target_regions)])
        
            else:
                cpg_res = ray.get([self.corncob_cpg.remote(fits[cpg],site_names[cpg]) for cpg in range(meth.shape[0])]) 
                ray.shutdown()
                cpg_res = pd.concat(cpg_res)
                if len(cometh) > 0:
                    cpg_res = cpg_res.sort_values('site')
                    cpg_res["site"] = np.repeat(self.unique(cometh),covs.shape[1])
                self.cpg_res = cpg_res
    
        else:
            
            if dmrs:
                cpg_res,region_res = zip(*[self.corncob_region_local(region,self.fits[self.target_regions==region],min_cpgs) for region in self.individual_regions])
                self.cpg_res = pd.concat(cpg_res)
                self.region_res = pd.concat(region_res)
                tmp=self.region_res[~self.region_res["site"].duplicated()]
                self.codistrib_regions = np.array([tmp[tmp["region"]==str(region)]["site"].iloc[next((i for i, sublist in enumerate(tmp[tmp["region"]==str(region)]["range"]) if pos in sublist ), -1)]
                 if any(int(pos) in reg for reg in tmp[tmp["region"]==str(region)]["range"]) else site_names[cpg] for pos,cpg,region in zip(self.region_cpg_indices,range(self.ncpgs), self.target_regions)])
        
            else:
                cpg_res = [self.corncob_cpg_local(fits[cpg],site_names[cpg]) for cpg in range(meth.shape[0])]
                cpg_res = pd.concat(cpg_res)
                if len(cometh) > 0:
                    cpg_res = cpg_res.sort_values('site')
                    cpg_res["site"] = np.repeat(self.unique(cometh),covs.shape[1])
                self.cpg_res = cpg_res

    def get_contrast(self,res: str="cpg",contrast: str="intercept",alpha=0.05,padj_method="fdr_bh"):
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
        sig = multipletests(res["p"], alpha, padj_method)
        with pd.option_context('mode.chained_assignment', None):
            res["padj"] = sig[1]
            res["sig"] = sig[0]
        return res

    @staticmethod
    def corncob_region_local(region,fits,min_cpgs=3):
        """
        Perform differential methylation analysis on a single target region using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
        mean and association with covariates are similar.
    
        Parameters:
            region (integer, float, or string): Name of the region being analysed.
            fits (1D numpy array): Array containing bb fits (corcon_2 object) for all cpgs in the region.
            min_cpgs (integer, default: 3): Minimum length of a dmr.
            
        Returns:
            pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
        """
    
        start=0
        end=2
        current_region=1
        region_res=[]
        res_1 = fits[start].waltdt()[0]
        cpg_res = [res_1] #Add individual cpg res to cpg result table
        X_c = pd.concat([fits[start].X,fits[start].X,fits[start].X]) 
        X_star_c = pd.concat([fits[start].X_star,fits[start].X_star,fits[start].X_star]) 
        X_c.index = range(X_c.shape[0])
        X_star_c.index = range(X_star_c.shape[0])
        
        while end < len(fits)+1:

            coord = np.s_[start:end] # Combined region to be tested

            res_2 = fits[end-1].waltdt()[0]
            cpg_res += [res_2] #Add individual cpg res to cpg result table
            
            if start+min_cpgs < len(fits)+1:
                ll_c = corncob_cpg_local("",np.hstack([fit.count for fit in fits[start:max(start+min_cpgs,end)]]),
                                             np.hstack([fit.total for fit in fits[start:max(start+min_cpgs,end)]]),X_c,X_star_c,region,LL=True,res=False)
                ll_s = np.array([fits[cpg].LogLike for cpg in range(start,max(start+min_cpgs,end))]).sum()
                
                bic_c = -2 * ll_c + ((fits[start].X.shape[1]+fits[start].X_star.shape[1])) * np.log(fits[start].X.shape[0]) 
                bic_s = -2 * (ll_s) + ((fits[start].X.shape[1]+fits[start].X_star.shape[1])*(max(start+min_cpgs,end)-start)) * np.log(fits[start].X.shape[0]) 
                end += 1
    
                #If sites come from the same distribution, keep extending the region
                if bic_c < bic_s:
                    ll_1=ll_c
                    previous_coord=coord
                    if end-start>=min_cpgs+1:
                        X_c = pd.concat([X_c,fits[start].X]) 
                        X_star_c = pd.concat([X_star_c,fits[start].X_star]) 
                        X_c.index = range(X_c.shape[0])
                        X_star_c.index = range(X_star_c.shape[0])
                    
                else: # Else start a new region
                    if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                        current_region+=1
                        region_res+=[corncob_cpg_local(f'{start}-{end-3}',np.vstack([fit.count for fit in fits[start:end-2]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end-2]]).sum(axis=0),
                                                       fits[start].X,fits[start].X_star,region)] #Add regions results to results table
                    start=end-2
                    X_c = pd.concat([fits[start].X,fits[start].X,fits[start].X]) 
                    X_star_c = pd.concat([fits[start].X_star,fits[start].X_star,fits[start].X_star]) 
                    X_c.index = range(X_c.shape[0])
                    X_star_c.index = range(X_star_c.shape[0])

            else:
                end += 1
    
        if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
            region_res+=[corncob_cpg_local(f'{start}-{end}',np.vstack([fit.count for fit in fits[start:end]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end]]).sum(axis=0),
                                           fits[start].X,fits[start].X_star,region)]
    
        cpg_res = pd.concat(cpg_res)
        if not region_res:
            region_res = None
        else:
            region_res = pd.concat(region_res)
            
        return cpg_res,region_res

    @staticmethod
    @ray.remote
    def corncob_region(region,region_id,fits_id,min_cpgs=3):
        """
        Perform differential methylation analysis on a single target region using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
        mean and association with covariates are similar.
    
        Parameters:
            region (integer, float, or string): Name of the region being analysed.
            region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array containing region assignment for all cpgs.
            fits (ray ID): ID of global value produced by ray.put(), pointing to a global array containing bb fits (corcon_2 object) for all cpgs.
            min_cpgs (integer, default: 3): Minimum length of a dmr.
            
        Returns:
            pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg in a region, with a seperate dataframe for region results if dmrs=True. 
        """
    
        fits=fits_id[region_id==region]
    
        start=0
        end=2
        current_region=1
        region_res=[]
        res_1 = fits[start].waltdt()[0]
        cpg_res = [res_1] #Add individual cpg res to cpg result table
        X_c = pd.concat([fits[start].X,fits[start].X,fits[start].X]) 
        X_star_c = pd.concat([fits[start].X_star,fits[start].X_star,fits[start].X_star]) 
        X_c.index = range(X_c.shape[0])
        X_star_c.index = range(X_star_c.shape[0])
        
        while end < len(fits)+1:

            coord = np.s_[start:end] # Combined region to be tested

            res_2 = fits[end-1].waltdt()[0]
            cpg_res += [res_2] #Add individual cpg res to cpg result table
            
            if start+min_cpgs < len(fits)+1:
                ll_c = corncob_cpg_local("",np.hstack([fit.count for fit in fits[start:max(start+min_cpgs,end)]]),
                                             np.hstack([fit.total for fit in fits[start:max(start+min_cpgs,end)]]),X_c,X_star_c,region,LL=True,res=False)
                ll_s = np.array([fits[cpg].LogLike for cpg in range(start,max(start+min_cpgs,end))]).sum()
                
                bic_c = -2 * ll_c + ((fits[start].X.shape[1]+fits[start].X_star.shape[1])) * np.log(fits[start].X.shape[0]) 
                bic_s = -2 * (ll_s) + ((fits[start].X.shape[1]+fits[start].X_star.shape[1])*(max(start+min_cpgs,end)-start)) * np.log(fits[start].X.shape[0]) 
                end += 1
    
                #If sites come from the same distribution, keep extending the region
                if bic_c < bic_s:
                    ll_1=ll_c
                    previous_coord=coord
                    if end-start>=min_cpgs+1:
                        X_c = pd.concat([X_c,fits[start].X]) 
                        X_star_c = pd.concat([X_star_c,fits[start].X_star]) 
                        X_c.index = range(X_c.shape[0])
                        X_star_c.index = range(X_star_c.shape[0])
                    
                else: # Else start a new region
                    if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                        current_region+=1
                        region_res+=[corncob_cpg_local(f'{start}-{end-3}',np.vstack([fit.count for fit in fits[start:end-2]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end-2]]).sum(axis=0),
                                                       fits[start].X,fits[start].X_star,region)] #Add regions results to results table
                    start=end-2
                    X_c = pd.concat([fits[start].X,fits[start].X,fits[start].X]) 
                    X_star_c = pd.concat([fits[start].X_star,fits[start].X_star,fits[start].X_star]) 
                    X_c.index = range(X_c.shape[0])
                    X_star_c.index = range(X_star_c.shape[0])

            else:
                end += 1
    
        if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
            region_res+=[corncob_cpg_local(f'{start}-{end}',np.vstack([fit.count for fit in fits[start:end]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end]]).sum(axis=0),
                                           fits[start].X,fits[start].X_star,region)]
    
        cpg_res = pd.concat(cpg_res)
        if not region_res:
            region_res = None
        else:
            region_res = pd.concat(region_res)
            
        return cpg_res,region_res

    @ray.remote
    def corncob_cpg(fit,site):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            fit (corncob_2 object): Betabinomial fit (corcon_2 object) for a cpg.
            site (integer, float, or string): Cpg row number in meth/coverage.
            
        Returns:
            pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
        """
        res = fit.waltdt()[0]
        res["site"] = f'{site}'
        return res

    @staticmethod
    def corncob_cpg_local(fit,site):
        """
        Perform differential methylation analysis on a single cpg using beta binomial regression.
    
        Parameters:
            fit (corncob_2 object): Betabinomial fit (corcon_2 object) for a cpg.
            site (integer, float, or string): Cpg row number in meth/coverage.
            
            
        Returns:
            pandas dataframe (optional, depending in res parameter): Dataframe containing estimates of coeficents for the cpg. 
            float (optional, depending in LL parameter): Model log likelihood
        """
        res = fit.waltdt()[0]
        res["site"] = f'{site}'
        return res

    def region_plot(self,region: int,contrast: str|list="",beta_vals: np.ndarray=np.array([]),show_codistrib_regions=True):
        """
        Plot methylation level within a region.
    
        Parameters:
            region (float, integer, or string): Name of this region.
            contrast (optional: string): Name of column in self.X (covs) or list denoting group membership. Groups methylation levels over a region will be plotted as seperate lines.
            beta_vals (optional: np.array): Optional array of beta values to plot if wishing to plot samples not in object (e.g. simulated samples). Must have equal number of cpgs (rows)
            to the meth and coverage data input to the object (object.ncpgs).
            show_codistrib_regions (bool, default: True): Whether to show codistributed regions as shaded areas on the plot.
        """

        if show_codistrib_regions:
            assert len(self.codistrib_regions) > 0, "Run bbseq with dmrs=True to find codistributed regions before setting show_codistrib_regions=True"
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
        region_plot["CpG"] = range(0,region_plot.shape[0])
        xlab = "CpG Number in Region"

        if show_codistrib_regions:
            assert len(self.codistrib_regions) == self.ncpgs, "Run bbseq before simulating if codistrib_regions=True"
            region_colours=["lightgrey","#5bd7f0", "lightgreen","#f0e15b", "#f0995b", "#db6b6b", "#cd5bf0"]
            codistrib_regions = self.codistrib_regions[self.target_regions == region]
            unique_codistrib_regions = self.unique([x for x in self.codistrib_regions[self.target_regions == region] if "_" in x])
            cdict = {x: region_colours[i] for i,x in enumerate(unique_codistrib_regions)}
            region_plot["region"] = [cdict[x] if "_" in x else None for x in codistrib_regions]
        if show_codistrib_regions:  
            region_plot=pd.melt(region_plot, id_vars=["CpG","region"],value_name='Beta Value')
        else:
            region_plot=pd.melt(region_plot, id_vars=["CpG"],value_name='Beta Value')
        if len(contrast) == 0:
            ax = sns.lineplot(region_plot, x="CpG", y="Beta Value")
        else:
            ax = sns.lineplot(region_plot, x="CpG", y="Beta Value", hue="variable")
        if show_codistrib_regions:
            ranges = region_plot.groupby('region')['CpG'].agg(['min', 'max'])
            for i, row in ranges.iterrows():
                ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=i, alpha=0.3)
            region_plot.loc[region_plot["region"].isna(),"region"] = "#4a4a4d"
            if len(contrast) == 0:
                sns.scatterplot(region_plot, x="CpG", y=means, c=region_plot.drop_duplicates("CpG")["region"], ax=ax)
            else:
                for group in self.unique(contrast):
                    sns.scatterplot(region_plot[region_plot["variable"]==self.unique(contrast)[0]], x="CpG", y=means[group], 
                                    c=region_plot[region_plot["variable"]==self.unique(contrast)[0]].drop_duplicates("CpG")["region"], ax=ax)
            ax.set_title(f"Codistributed Regions in Region {region}")
        else:
            if len(contrast) == 0:
                sns.scatterplot(region_plot, x="CpG", y=means, ax=ax)
            else:
                for group in self.unique(contrast):
                    sns.scatterplot(region_plot[region_plot["variable"]==self.unique(contrast)[0]], x="CpG", y=means[group], ax=ax)
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

    def sum_regions(self,regions,array):
        """
        Aggregates array values within predefined regions.
    
        Parameters:
            regions (numpy array): Regions each column of array belongs to (p-dimensional vector).
            array (numpy array): Array to be summarised.
            return_order (boolean, default: False): Whether to return the order of regions in the summarised array (can change from regions).
            
        Returns:
            numpy array: Summarised Array.
        """
        unique_regions, inverse_indices = np.unique(regions, return_inverse=True)
        group_sums = np.zeros((len(unique_regions), array.shape[1]), dtype='int')
        # Accumulate column sums within groups
        np.add.at(group_sums, inverse_indices, array)
        return group_sums[self.unique(inverse_indices),:]