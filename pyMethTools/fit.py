import numpy as np
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta
import pandas as pd
from itertools import chain
from functools import partial
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns

def fit_betabinom(meth, coverage, target_regions,chunksize=1,ncpu=1,maxiter=250,maxfev=250):
    """
    Fit beta binomial model to DNA methylation data.

    Parameters:
        meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
        chunksize (integer, default: 1): Number of regions to process at once if using parallel processing.
        ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        numpy array: Array of optimisation results of length equal to the number of cpgs. 
    """

    individual_regions=unique(target_regions)
    
    if ncpu > 1: #Use ray for parallel processing
        ray.init(num_cpus=ncpu)    
        meth_id=ray.put(meth)
        coverage_id=ray.put(coverage)
        region_id=ray.put(target_regions)
        if chunksize>1:
            result = ray.get([fit_betabinom_chunk.remote(individual_regions[chunk:chunk+chunksize],
                                                     meth_id,coverage_id,region_id,maxiter,maxfev) 
                          for chunk in range(0,len(set(individual_regions))+1,chunksize)])
            result = list(chain.from_iterable(result))
        else:
            result = ray.get([fit_betabinom_region.remote(region,meth_id,coverage_id,region_id,maxiter,maxfev) 
                          for region in individual_regions])
        ray.shutdown()
        
    else:
        result = [fit_betabinom_region_local(region,meth[target_regions==region],coverage[target_regions==region],
                                             target_regions[target_regions==region],maxiter,maxfev,sequential=True) 
                          for region in individual_regions]
        
    result = np.array(list(chain.from_iterable(result)))
        
    return result

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
    
    individual_regions=set(chunk_regions)
    chunk_result = [fit_betabinom_region_local(region,meth_id,coverage_id,region_id,maxiter,maxfev) 
                    for region in individual_regions]
    return chunk_result

def fit_betabinom_region_local(region,meth_id,coverage_id,region_id,maxiter=250,maxfev=250,sequential=False):
    """
    Fit beta binomial model to cpgs within a single target region.

    Parameters:
        region (float, integer, or string): Name of this region.
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        sequential (boolean, default: False): Whether this is being run in sequential mode, or within a parallel backend. If true expects all id variabled to be actual objects.
        
    Returns:
        numpy array: Array of optimisation results of length equal to the number of cpgs in this region. 
    """
    
    if sequential:
        region_meth=meth_id
        region_coverage=coverage_id
    else:
        region_meth=meth_id[region_id==region]
        region_coverage=coverage_id[region_id==region]

    #Get priors based on region average
    region_prior=fit_betabinom_cpg(region_meth.mean(axis=0),region_coverage.mean(axis=0),[15,30],maxiter,maxfev).x
    inits=[region_prior[0],region_prior[1]]

    region_fits = [fit_betabinom_cpg(region_meth[cpg,:],region_coverage[cpg,:],inits,maxiter,maxfev) for cpg in range(region_meth.shape[0])]

    return region_fits

@ray.remote
def fit_betabinom_region(region,meth_id,coverage_id,region_id,maxiter=250,maxfev=250):
    """
    Fit beta binomial model to cpgs within a single target region.

    Parameters:
        region (float, integer, or string): Name of this region.
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        numpy array: Array of optimisation results from scipy minimize of length equal to the number of cpgs in this region. 
    """
    
    region_meth=meth_id[region_id==region]
    region_coverage=coverage_id[region_id==region]

    #Get priors based on region average
    region_prior=fit_betabinom_cpg(region_meth.mean(axis=0),region_coverage.mean(axis=0),[15,30],maxiter,maxfev).x
    inits=[region_prior[0],region_prior[1]]

    region_fits = [fit_betabinom_cpg(region_meth[cpg,:],region_coverage[cpg,:],inits,maxiter,maxfev) for cpg in range(region_meth.shape[0])]

    return region_fits

def fit_betabinom_cpg(cpg_meth,cpg_coverage,inits,maxiter=250,maxfev=250):
    """
    Fit beta binomial model to a single cpg.

    Parameters:
        cpg_meth (1D numpy array): Array of methylated reads at the cpg for each sample.
        cpg_coverage (1D numpy array): Array of total reads at the cpg for each sample.
        inits (1D numpy array): Initial alpha and beta paramters for the beta binomial model.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        scipy.optimize._optimize.OptimizeResult: Optimisation results from scipy minimize.
    """
    return minimize(bbll,inits,args=(cpg_meth,cpg_coverage),method='Nelder-Mead',options={"maxiter":maxiter,"maxfev":maxfev},bounds=Bounds(0,np.inf))

    # minimize negative log likelihood
def bbll(params,k,n):
    """
    Compute the negative log-likelihood of a beta-binomial model.

    Parameters:
        params (list): alpha and beta parameters of the model.
        k (1D numpy array): Array of methylated reads at the cpg for each sample.
        n (1D numpy array): Array of total reads at the cpg for each sample.
        
    Returns:
        float: Negative log-likelihood.
    """
    a,b=params
    ll = betabinom.logpmf(k,n,a,b)
    return -ll.sum()

# Define the log-likelihood for beta-binomial regression (simplified version)
def beta_binomial_log_likelihood(self, beta):
    """
    Compute the negative log-likelihood for beta-binomial regression.

    Parameters:
        beta (numpy array): Coefficients to estimate (p-dimensional vector).
        
    Returns:
        float: Negative log-likelihood.
    """
    mu_wlink = np.matmul(
            self.X,
            beta[:self.n_param_abd]
        )
    phi_wlink = np.matmul(
            self.X_star,
            beta[-1*self.n_param_disp:]
        )
    mu = expit(beta[0])
    phi = expit(beta[0]) 

    a = mu*(1-phi)/phi
    b = (1-mu)*(1-phi)/phi
    LL = stats.betabinom.logpmf(
            k=self.count,
            n=self.total,
            a=a,
            b=b
        )

    return -1*np.sum(LL)  # Negative for minimization

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