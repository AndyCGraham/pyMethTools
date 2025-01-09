import numpy as np
from scipy.optimize import minimize,Bounds
from scipy.stats import betabinom
from scipy.special import gammaln,binom,beta
import pandas as pd
import random
from itertools import chain
from functools import partial
import math
import ray
from ray.util.multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from pyMethTools.fit import fit_betabinom_cpg,unique

def find_comethyl_regions(fits,meth,coverage,target_regions,min_cpgs=3,ncpu=1,chunksize=1,maxiter=250,maxfev=250):
    """
    Find regions of contigous cpgs which likely arise from the same beta binomial distribution.

    Parameters:
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs. 
        meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
        min_cpgs (integer, default: 3): Minimum length of a codistributed region.
        ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
        chunksize (integer, default: 1): Number of regions to process at once if using parallel processing.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        1D numpy array: Array containing a value for each cpg specifying its cmethylated region or individual cpg assignment. 
    """
    
    individual_regions=unique(target_regions)
    
    if ncpu > 1: #Use ray for parallel processing
        ray.init(num_cpus=ncpu)    
        fits_id=ray.put(fits)
        meth_id=ray.put(meth)
        coverage_id=ray.put(coverage)
        region_id=ray.put(target_regions)
        if chunksize>1:
            regions = ray.get([comethyl_regions_chunk.remote(individual_regions[chunk:chunk+chunksize],
                                                     fits[np.isin(target_regions,individual_regions[chunk:chunk+chunksize])],
                                                            meth_id,coverage_id,region_id,min_cpgs,
                                                            maxiter,maxfev) 
                          for chunk in range(0,len(set(individual_regions))+1,chunksize)])
            regions = list(chain.from_iterable(result))
        else:
            regions = ray.get([comethyl_regions.remote(region,fits[target_regions==region],meth_id,coverage_id,region_id,min_cpgs,
                                                      maxiter,maxfev) for region in individual_regions])
        ray.shutdown()

    else:
        regions = [comethyl_regions_local(region,fits[target_regions==region],meth[target_regions==region],
                                          coverage[target_regions==region],
                  target_regions[target_regions==region],min_cpgs,maxiter,maxfev,sequential=True) 
                   for region in individual_regions]

    regions = np.hstack(regions)

    return regions

@ray.remote
def comethyl_regions_chunk(chunk_regions,fits,meth_id,coverage_id,region_id,min_cpgs=3,maxiter=250,maxfev=250):
    """
    Find regions of contigous cpgs which likely arise from the same beta binomial distribution, in a chunk of target regions from DNA methylation data.

    Parameters:
        chunk_regions (numpy array): Array of region assignments for this chunk.
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in the chunk regions. 
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
        min_cpgs (integer, default: 3): Minimum length of a codistributed region.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        1D numpy array: Array containing a value for each cpg specifying its cmethylated region or individual cpg assignment, for cpgs in this chunk.  
    """
    
    individual_regions=unique(chunk_regions)
    chunk_result = [comethyl_regions_local(region,fits[region_id==region],meth_id,coverage_id,region_id,min_cpgs,maxiter,maxfev) 
                    for region in individual_regions]
    return chunk_result

def comethyl_regions_local(region,fits,meth_id,coverage_id,region_id,min_cpgs=3,maxiter=250,maxfev=250,sequential=False):
    """
    Find regions of contigous cpgs which likely arise from the same beta binomial distribution, within a target region.

    Parameters:
        region (float, integer, or string): Name of this region.
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in this region.
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
        min_cpgs (integer, default: 3): Minimum length of a codistributed region.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        sequential (boolean, default: False): Whether this is being run in sequential mode, or within a parallel backend. If true expects all id variabled to be actual objects.
        
    Returns:
        1D numpy array: Array containing a value for each cpg specifying its cmethylated region or individual cpg assignment of length equal to the number of cpgs in this region. 
    """
    
    if sequential:
        meth=meth_id
        coverage=coverage_id
    else:
        meth=meth_id[region_id==region]
        coverage=coverage_id[region_id==region]
        
    start=0
    end=2
    current_region=1
    fit1=fits[0]
    regions=np.repeat(region,meth.shape[0]).astype(np.float32)
    ll_1 = betabinom.logpmf(meth[0,:],coverage[0,:],fit1.x[0],fit1.x[1]).sum()
    
    while end < meth.shape[0]+1:
        coord = np.s_[start:end] # Combined region to be tested
        
        combined = fit_betabinom_cpg(meth[coord,:].flatten(),coverage[coord,:].flatten(),[fit1.x[0],fit1.x[1]],maxiter,maxfev)
        ll_c = betabinom.logpmf(meth[coord,:].flatten(),coverage[coord,:].flatten(),combined.x[0],combined.x[1]).sum()
        ll_2 = betabinom.logpmf(meth[end-1,:],coverage[end-1,:],fits[end-1].x[0],fits[end-1].x[1]).sum()

        bic_c = 3*math.log((end-start)*meth.shape[1]) - 2*ll_c
        bic_s = 6*math.log((end-start)*meth.shape[1]) - 2*(ll_1+ll_2)
        end += 1

        #If sites come from the same distribution, keep extending the region
        if bic_c < bic_s:
            fit1=combined
            ll_1=ll_c
            previous_coord=coord
        else: # Else start a new region
            if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                regions[start:end-2] = f"{region}.{current_region}"
                current_region+=1
            start=end-2
            ll_1=ll_2

    if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
        regions[start:end] = f"{region}.{current_region}"

    return regions

@ray.remote
def comethyl_regions(region,fits,meth_id,coverage_id,region_id,min_cpgs=3,maxiter=250,maxfev=250):
    """
    Find regions of contigous cpgs which likely arise from the same beta binomial distribution, within a target region.

    Parameters:
        region (float, integer, or string): Name of this region.
        fits (1D numpy array): Array of optimisation results of length equal to the number of cpgs in this region.
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        region_id (ray ID): ID of global value produced by ray.put(), pointing to a global array of cpg region assignments.
        min_cpgs (integer, default: 3): Minimum length of a codistributed region.
        maxiter (integer, default: 250): Maxinum number of iterations when fitting beta binomial model to a cpg (see numpy minimize).
        maxfev (integer, default: 250): Maxinum number of evaluations when fitting beta binomial model to a cpg (see numpy minimize).
        
    Returns:
        1D numpy array: Array containing a value for each cpg in this region specifying its cmethylated region or individual cpg assignment. 
    """
    
    meth=meth_id[region_id==region]
    coverage=coverage_id[region_id==region]
    start=0
    end=2
    current_region=1
    fit1=fits[0]
    regions=np.empty((meth.shape[0], 2))
    regions[:,0] = region
    regions[:,1] = np.nan
    ll_1 = betabinom.logpmf(meth[0,:],coverage[0,:],fit1.x[0],fit1.x[1]).sum()
    
    start=0
    end=2
    current_region=1
    fit1=fits[0]
    regions=np.repeat(region,meth.shape[0]).astype(np.float32)
    ll_1 = betabinom.logpmf(meth[0,:],coverage[0,:],fit1.x[0],fit1.x[1]).sum()
    
    while end < meth.shape[0]+1:
        coord = np.s_[start:end] # Combined region to be tested
        
        combined = fit_betabinom_cpg(meth[coord,:].flatten(),coverage[coord,:].flatten(),[fit1.x[0],fit1.x[1]],maxiter,maxfev)
        ll_c = betabinom.logpmf(meth[coord,:].flatten(),coverage[coord,:].flatten(),combined.x[0],combined.x[1]).sum()
        ll_2 = betabinom.logpmf(meth[end-1,:],coverage[end-1,:],fits[end-1].x[0],fits[end-1].x[1]).sum()

        bic_c = 3*math.log((end-start)*meth.shape[1]) - 2*ll_c
        bic_s = 6*math.log((end-start)*meth.shape[1]) - 2*(ll_1+ll_2)
        end += 1

        #If sites come from the same distribution, keep extending the region
        if bic_c < bic_s:
            fit1=combined
            ll_1=ll_c
            previous_coord=coord
        else: # Else start a new region
            if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                regions[start:end-2] = f"{region}.{current_region}"
                current_region+=1
            start=end-2
            ll_1=ll_2

    if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
        regions[start:end] = f"{region}.{current_region}"

    return regions
    
def region_plot(region_num,beta_vals,cometh,groups=[]):
    """
    Plot methylation level within a region.

    Parameters:
        region_num (float, integer, or string): Name of this region.
        beta_vals (2D numpy array): Table of proportion methylated reads (beta values) at each cpg (rows) for each sample (columns).
        cometh (1D numpy array): Array of same length as number of cpgs in beta_vals, specifying which comethylated region each cpg belongs to.
        groups (optional: 1D numpy array): Array of same length as number of samples in beta_vals, specifying an experimental group (e.g. treated/untreated) 
        each sample belongs to. Groups methylation levels over a region will be plotted as seperate lines.
    """
    
    region = cometh[np.floor(cometh) == region_num]
    region_plot = pd.DataFrame(beta_vals[np.floor(cometh) == region_num])
    
    if len(groups) > 0:
        region_plot.columns=groups
        means = {group: region_plot.loc[:,groups==group].mean(axis=1) for group in unique(groups)}
    else:
        means = region_plot.mean(axis=1)
    region_plot["CpG"] = range(0,region_plot.shape[0])
    xlab = "CpG Number in Region"
        
    region_colours=["lightgrey","#5bd7f0", "lightgreen","#f0e15b", "#f0995b", "#db6b6b", "#cd5bf0"]
    region_plot.loc[:,"region"] = [region_colours[round(num)-1] if num !=0 else None for num in (region % 1)*10]
    region_plot=pd.melt(region_plot, id_vars=["CpG","region"],value_name='Beta Value')
    if len(groups) == 0:
        ax = sns.lineplot(region_plot, x="CpG", y="Beta Value")
    else:
        ax = sns.lineplot(region_plot, x="CpG", y="Beta Value", hue="variable")
    ranges = region_plot.groupby('region')['CpG'].agg(['min', 'max'])
    for i, row in ranges.iterrows():
        ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=i, alpha=0.3)
    region_plot.loc[region_plot["region"].isna(),"region"] = "#4a4a4d"
    if len(groups) == 0:
        sns.scatterplot(region_plot, x="CpG", y=means, c=region_plot.drop_duplicates("CpG")["region"], ax=ax)
    else:
        for group in unique(groups):
            sns.scatterplot(region_plot[region_plot["variable"]==unique(groups)[0]], x="CpG", y=means[group], 
                            c=region_plot[region_plot["variable"]==unique(groups)[0]].drop_duplicates("CpG")["region"], ax=ax)
    
    ax.set_title(f"Comethylated Regions in Twist Region {region_num}")
    plt.xlabel(xlab)
