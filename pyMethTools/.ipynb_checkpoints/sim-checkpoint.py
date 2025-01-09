import numpy as np
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta
from itertools import chain
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pyMethTools.fit import unique

def sim_multiple_cpgs(fits,target_regions=[],cometh_regions=False,read_depth=30,vary_read_depth=True,read_depth_sd=5,adjust_factor=0,
                      diff_regions=[],n_diff_regions=0,sample_size=100,ncpu=1,chunksize=1):
    """
    Simulate new samples based on existing samples.

    Parameters:
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs. 
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
        cometh (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which comethylated region each cpg belongs to.
        read_depth (integer, default: 30): Desired average read depth of simulated samples.
        vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
        read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
        adjust_factor (float or integer, default: 0): The log2FoldChange of percent methylation, at 'diff_regions' or in 'n_diff_regions' in simulated samples.
        0 will not alter methylation levels. By default, the log2FoldChange will be positive and negative in half the regions affected.
        diff_regions (List or numpy array, default: []): Names of regions to be affected by 'adjust_factor'.
        n_diff_regions (integer, default: 0): If diff_regions not specified, the number of regions to be affected by 'adjust_factor'.
        sample_size (integer, default: 100): Number of samples to simulate.
        ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
        chunksize (integer, default: 1): Number of regions to process at once if using parallel processing.
        
    Returns:
        2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
    """

    if len(target_regions)==0:
        target_regions=np.floor(cometh_regions)
        
    individual_regions=unique(target_regions)
    adjust_factors = np.repeat(0.0, len(fits))
    if len(diff_regions)>0 or n_diff_regions>0:
        if len(diff_regions)>0:
            diff_regions_up = random.sample(diff_regions,int(len(diff_regions)/2))
            diff_regions_down = diff_regions[~np.isin(cometh_regions,diff_regions_up)]
        elif(n_diff_regions>0):
            independent_regions = np.array([region for region in unique(cometh_regions) if region % 1 != 0])
            print(independent_regions)
            diff_regions_up = np.random.choice(independent_regions,int(n_diff_regions/2))
            diff_regions_down = np.random.choice(independent_regions[~np.isin(independent_regions,diff_regions_up)],int(n_diff_regions/2))
        adjust_factors[np.isin(cometh_regions,diff_regions_up)] = adjust_factor
        adjust_factors[np.isin(cometh_regions,diff_regions_down)] = -adjust_factor

    else:
        adjust_factors = np.repeat(0,len(fits))
    
    if ncpu > 1: #Use ray for parallel processing
        ray.init(num_cpus=ncpu)    
        if chunksize>1:
            sim_meth, sim_coverage = zip(*ray.get([sim_chunk.remote(fits[np.isin(target_regions,individual_regions[chunk:chunk+chunksize])],target_regions[np.isin(target_regions,individual_regions[chunk:chunk+chunksize])],individual_regions[chunk:chunk+chunksize],read_depth,vary_read_depth,read_depth_sd, adjust_factors[np.isin(target_regions,individual_regions[chunk:chunk+chunksize])],sample_size) for chunk in range(0,len(individual_regions)+1,chunksize)]))
            #sim_meth, sim_coverage = list(chain.from_iterable(sims))
        else:
            sim_meth, sim_coverage = zip(*ray.get([sim.remote(fits[target_regions==region],read_depth,vary_read_depth,read_depth_sd,adjust_factors[target_regions==region],sample_size) for region in individual_regions]))
        ray.shutdown()

    else:
        sim_meth, sim_coverage = zip(*[sim_local(fits[target_regions==region],read_depth,vary_read_depth,read_depth_sd,adjust_factors[target_regions==region],sample_size) for region in individual_regions])
    return np.vstack(sim_meth), np.vstack(sim_coverage), adjust_factors

@ray.remote
def sim_chunk(fits,target_regions,individual_regions,read_depth=30,vary_read_depth=True,read_depth_sd=5,adjust_factors=0,sample_size=100):
    """
    Simulate new samples based on existing samples, for this chunk of regions.

    Parameters:
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in this chunk of regions. 
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg in this chunk belongs to.
        individual_regions (optional: 1D numpy array): Array of unique 'target_regions' in this chunk.
        read_depth (integer, default: 30): Desired average read depth of simulated samples.
        vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
        read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
        adjust_factors (1D numpy array, default: 0): Array of the log2FoldChange each cpg will be altered by in the simulated versus template samples.
        sample_size (integer, default: 100): Number of samples to simulate.
        
    Returns:
        2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
    """
    sim_meth_chunk, sim_coverage_chunk = zip(*[sim_local(fits[target_regions==region],read_depth,vary_read_depth,read_depth_sd,adjust_factors[target_regions==region],sample_size) for region in individual_regions])
    return sim_meth_chunk, sim_coverage_chunk

def sim_local(fits,read_depth=30,vary_read_depth=True,read_depth_sd=5,adjust_factors=0,sample_size=100):
    """
    Simulate new samples based on existing samples, for this region.

    Parameters:
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in this region. 
        read_depth (integer, default: 30): Desired average read depth of simulated samples.
        vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
        read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
        adjust_factors (1D numpy array, default: 0): Array of the log2FoldChange each cpg will be altered by in the simulated versus template samples.
        sample_size (integer, default: 100): Number of samples to simulate.
        
    Returns:
        2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
    """
    meth=np.empty((len(fits),sample_size),dtype=int)
    coverage=np.empty((len(fits),sample_size ),dtype=int)

    if vary_read_depth:
        for sample in range(sample_size):
            coverage[:,sample] = int(np.random.normal(read_depth,read_depth_sd))
    else:
        coverage.fill(read_depth)
        
    for cpg in range(len(fits)):
        if adjust_factors[cpg] != 0:
            mu=fits[cpg].x[0]/(fits[cpg].x[0]+fits[cpg].x[1])
            mu=mu*(2**adjust_factors[cpg])
            mu = max(min(mu,1),0)
            phi=1/(fits[cpg].x[0]+fits[cpg].x[1]+1)
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        else:
            a=fits[cpg].x[0]
            b=fits[cpg].x[1]
        for sample in range(sample_size):
            try:
                meth[cpg,sample] = betabinom.rvs(coverage[cpg,sample],a,b,size=1)[0]
            except:
                raise ValueError("Parameters are not compatible with the beta binomial model. Likely adjust_factor is too high or something went wrong during fitting.")
    return meth, coverage

@ray.remote    
def sim(fits,read_depth=30,vary_read_depth=True,read_depth_sd=5,adjust_factors=0,sample_size=100):
    """
    Simulate new samples based on existing samples, for this region.

    Parameters:
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in this region. 
        read_depth (integer, default: 30): Desired average read depth of simulated samples.
        vary_read_depth (boolean, default: True): Whether to vary sample read depth per sample (if false all coverage values will equal 'read_depth')
        read_depth_sd (float or integer, default: 5): The standard deviation to vary read depth by, using a normal distribution with mean 'read_depth'.
        adjust_factors (1D numpy array, default: 0): Array of the log2FoldChange each cpg will be altered by in the simulated versus template samples.
        sample_size (integer, default: 100): Number of samples to simulate.
        
    Returns:
        2D numpy array,2D numpy array: Arrays containing simulated methylated counts, total counts for simulated samples. 
    """
    meth=np.empty((len(fits),sample_size),dtype=int)
    coverage=np.empty((len(fits),sample_size ),dtype=int)

    if vary_read_depth:
        for sample in range(sample_size):
            coverage[:,sample] = int(np.random.normal(read_depth,read_depth_sd))
    else:
        coverage.fill(read_depth)
        
    for cpg in range(len(fits)):
        if adjust_factors[cpg] != 0:
            mu=fits[cpg].x[0]/(fits[cpg].x[0]+fits[cpg].x[1])
            mu=mu*(2**adjust_factors[cpg])
            mu = max(min(mu,1),0)
            phi=1/(fits[cpg].x[0]+fits[cpg].x[1]+1)
            a = mu*(1-phi)/phi
            b = (1-mu)*(1-phi)/phi
        else:
            a=fits[cpg].x[0]
            b=fits[cpg].x[1]
        for sample in range(sample_size):
            try:
                meth[cpg,sample] = betabinom.rvs(coverage[cpg,sample],a,b,size=1)[0]
            except:
                raise ValueError("Parameters are not compatible with the beta binomial model. Likely adjust_factor is too high or something went wrong during fitting.")
    return meth, coverage