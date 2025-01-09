from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from scipy import linalg
from scipy.stats import t,rankdata
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit,expit, digamma, polygamma
from pyMethTools.fit import unique
import ray
import warnings

def bbseq(meth,coverage,target_regions=[],cometh=[],site_names=[],covs=None,covs_disp=None,dmrs=True,min_cpgs=3,ncpu=1):
    """
    Perform differential methylation analysis using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
    mean and association with covariates are similar.

    Parameters:
        meth (2D numpy array): Count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage (2D numpy array): Count table of total reads at each cpg (rows) for each sample (columns).
        target_regions (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which target region each cpg belongs to.
        cometh (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying which comethylated region each cpg belongs to.
        site_names (optional: 1D numpy array): Array of same length as number of cpgs in meth/coverage, specifying names of each cpg site.
        covs (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the mean parameter, 
        with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column.
        covs_disp (optional: pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
        with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column
        dmrs (boolean, default: True): Whether to identify and test differentially methylated regions of contigous cpgs whose mean and association with covariates are similar.
        min_cpgs (integer, default: 3): Minimum length of a dmr.
        ncpu (integer, default: 1): Number of cpus to use. If > 1 will use a parallel backend with Ray.
        
    Returns:
        pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
    """

    individual_regions=unique(target_regions)
    
    if len(cometh) > 0:
        cometh = [region if region % 1 != 0 else f'{region}0{index}' for index,region in enumerate(cometh)]
        meth,site_names = sum_regions(cometh, meth, return_order=True) 
        coverage = sum_regions(cometh, coverage) 
        site_names = unique(site_names)
    elif not site_names:
        site_names = list(range(meth.shape[0]))
    
    if not isinstance(covs, pd.DataFrame):
        covs = pd.DataFrame(np.repeat(1,meth.shape[1]))
        covs.columns=['intercept']
        
    if not isinstance(covs_disp, pd.DataFrame):
        covs_disp = pd.DataFrame(np.repeat(1,meth.shape[0]))
        covs_disp.columns=['intercept']

    if ncpu > 1:
    
        ray.init(num_cpus=ncpu)
        meth_id=ray.put(meth)
        coverage_id=ray.put(coverage)
        region_id=ray.put(target_regions)
        X_id=ray.put(covs)
        X_star_id=ray.put(covs_disp)
        site_names_id=ray.put(site_names)
    
        if dmrs:
            cpg_res,region_res = zip(*ray.get([corncob_region.remote(region,region_id,meth_id,coverage_id,X_id,X_star_id,min_cpgs) for region in individual_regions]))
            ray.shutdown()
            cpg_res = pd.concat(cpg_res)
            region_res = pd.concat(region_res)
            return cpg_res,region_res
    
        else:
            cpg_res = ray.get([corncob_cpg.remote(cpg,meth_id,coverage_id,X_id,X_star_id,site_names_id) for cpg in range(meth.shape[0])])
            ray.shutdown()
            cpg_res = pd.concat(cpg_res)
            if len(cometh) > 0:
                cpg_res = cpg_res.sort_values('site')
                cpg_res["site"] = np.repeat(unique(cometh),covs.shape[1])
            cpg_res
            return cpg_res

    else:
        
        if dmrs:
            cpg_res,region_res = zip(*[corncob_region_local(region,meth[target_regions==region,:],coverage[target_regions==region,:],covs,covs_disp,min_cpgs) for region in individual_regions])
            cpg_res = pd.concat(cpg_res)
            region_res = pd.concat(region_res)
            return cpg_res,region_res
    
        else:
            cpg_res = [corncob_cpg_local(cpg,meth[cpg,:],coverage[cpg,:],covs,covs_disp,site_names[cpg]) for cpg in range(meth.shape[0])]
            cpg_res = pd.concat(cpg_res)
            if len(cometh) > 0:
                cpg_res = cpg_res.sort_values('site')
                cpg_res["site"] = np.repeat(unique(cometh),covs.shape[1])
            cpg_res
            return cpg_res

def get_contrast(res,contrast,alpha=0.05,padj_method="fdr_bh"):
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
    res = res[res.index == contrast]
    sig = multipletests(res["p"], alpha, padj_method)
    with pd.option_context('mode.chained_assignment', None):
        res["padj"] = sig[1]
        res["sig"] = sig[0]
    return res
    

def sum_regions(regions,array,return_order=False):
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
    if return_order:
        return group_sums,inverse_indices
    else:
        return group_sums

def corncob_region_local(region,meth,coverage,X,X_star,min_cpgs=3):
    """
    Perform differential methylation analysis on a single target region using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
    mean and association with covariates are similar.

    Parameters:
        region (integer, float, or string): Name of the region being analysed.
        meth (2D numpy array): Count table of methylated reads at each cpg in the region (rows) for each sample (columns).
        coverage (2D numpy array): Count table of total reads at each cpg in the region (rows) for each sample (columns).
        X (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the mean parameter, 
        with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column.
        X_star (pandas dataframe): Dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion parameter, 
        with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column.
        min_cpgs (integer, default: 3): Minimum length of a dmr.
        
    Returns:
        pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg, with a seperate dataframe for region results if dmrs=True. 
    """

    start=0
    end=2
    current_region=1
    region_res=[]
    regions=np.repeat(region,meth.shape[0]).astype(np.float32)
    res_1,ll_1 = corncob_cpg_local(start,meth[0,:],coverage[0,:],X,X_star,region,LL=True) 
    cpg_res = [res_1] #Add individual cpg res to cpg result table
    X_c = pd.concat([X,X]) 
    X_star_c = pd.concat([X_star,X_star]) 
    X_c.index = range(X_c.shape[0])
    X_star_c.index = range(X_star_c.shape[0])
    
    while end < meth.shape[0]+1:

        coord = np.s_[start:end] # Combined region to be tested
        
        ll_c = corncob_cpg_local("",meth[start:end,:].flatten(),coverage[start:end,:].flatten(),X_c,X_star_c,region,LL=True,res=False)
        res_2,ll_2 = corncob_cpg_local(end-1,meth[end-1,:],coverage[end-1,:],X,X_star,region,LL=True) 
        cpg_res += [res_2] #Add individual cpg res to cpg result table

        bic_c = -2 * ll_c + ((X.shape[1]+X_star.shape[1])) * np.log(X.shape[0]) 
        bic_s = -2 * (ll_1+ll_2) + ((X.shape[1]+X_star.shape[1])*2) * np.log(X.shape[0]) 
        end += 1

        #If sites come from the same distribution, keep extending the region
        if bic_c < bic_s:
            ll_1=ll_c
            previous_coord=coord
            X_c = pd.concat([X_c,X]) 
            X_star_c = pd.concat([X_star_c,X_star]) 
            X_c.index = range(X_c.shape[0])
            X_star_c.index = range(X_star_c.shape[0])
        else: # Else start a new region
            if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                current_region+=1
                region_res+=[corncob_cpg_local(f'{start}-{end-3}',meth[start:end-2,:].sum(axis=0),coverage[start:end-2,:].sum(axis=0),X,X_star,region)] #Add regions results to results table
            start=end-2
            ll_1=ll_2
            X_c = pd.concat([X,X]) 
            X_star_c = pd.concat([X_star,X_star]) 
            X_c.index = range(X_c.shape[0])
            X_star_c.index = range(X_star_c.shape[0])

    if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
        region_res+=[corncob_cpg_local(f'{start}-{end}',meth[start:end,:].sum(axis=1),coverage[start:end,:].sum(axis=1),X,X_star,region)]

    cpg_res = pd.concat(cpg_res)
    if not region_res:
        region_res = None
    else:
        region_res = pd.concat(region_res)
        
    return cpg_res,region_res
    
@ray.remote
def corncob_region(region,region_id,meth_id,coverage_id,X_id,X_star_id,min_cpgs=3):
    """
    Perform differential methylation analysis on a single target region using beta binomial regression, with an option to compute differentially methylated regions of contigous cpgs whose
    mean and association with covariates are similar.

    Parameters:
        region (integer, float, or string): Name of the region being analysed.
        meth_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of methylated reads at each cpg (rows) for each sample (columns).
        coverage_id (ray ID): ID of global value produced by ray.put(), pointing to a global count table of total reads at each cpg (rows) for each sample (columns).
        X_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design 
        matrix for the mean parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column.
        X_star_id (ray ID): ID of global value produced by ray.put(), pointing to a global dataframe shape (number samples, number covariates) specifying the covariate design matrix for the dispersion 
        parameter, with a seperate column for each covariate (catagorical covariates should be one-hot encoded), including intercept (column of 1's named 'intercept').
        If not supplied will form only an intercept column.
        min_cpgs (integer, default: 3): Minimum length of a dmr.
        
    Returns:
        pandas dataframe(s): Dataframe containing estimates of coeficents for each covariate for each cpg in a region, with a seperate dataframe for region results if dmrs=True. 
    """

    meth=meth_id[region_id==region]
    coverage=coverage_id[region_id==region]

    start=0
    end=2
    current_region=1
    region_res=[]
    regions=np.repeat(region,meth.shape[0]).astype(np.float32)
    res_1,ll_1 = corncob_cpg_local(start,meth[0,:],coverage[0,:],X_id,X_star_id,region,LL=True) 
    cpg_res = [res_1] #Add individual cpg res to cpg result table
    X_c = pd.concat([X_id,X_id]) 
    X_star_c = pd.concat([X_star_id,X_star_id]) 
    X_c.index = range(X_c.shape[0])
    X_star_c.index = range(X_star_c.shape[0])
    
    while end < meth.shape[0]+1:

        coord = np.s_[start:end] # Combined region to be tested
        
        ll_c = corncob_cpg_local("",meth[start:end,:].flatten(),coverage[start:end,:].flatten(),X_c,X_star_c,region,LL=True,res=False)
        res_2,ll_2 = corncob_cpg_local(end-1,meth[end-1,:],coverage[end-1,:],X_id,X_star_id,region,LL=True) 
        cpg_res += [res_2] #Add individual cpg res to cpg result table

        bic_c = -2 * ll_c + ((X.shape[1]+X_star.shape[1])) * np.log(X.shape[0]) 
        bic_s = -2 * (ll_1+ll_2) + ((X.shape[1]+X_star.shape[1])*2) * np.log(X.shape[0]) 
        end += 1

        #If sites come from the same distribution, keep extending the region
        if bic_c < bic_s:
            ll_1=ll_c
            previous_coord=coord
            X_c = pd.concat([X_c,X_id]) 
            X_star_c = pd.concat([X_star_c,X_star_id]) 
            X_c.index = range(X_c.shape[0])
            X_star_c.index = range(X_star_c.shape[0])
        else: # Else start a new region
            if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                current_region+=1
                region_res+=[corncob_cpg_local(f'{start}-{end-3}',meth[start:end-2,:].sum(axis=0),coverage[start:end-2,:].sum(axis=0),X,X_star,region)] #Add regions results to results table
            start=end-2
            ll_1=ll_2
            X_c = pd.concat([X_id,X_id]) 
            X_star_c = pd.concat([X_star_id,X_star_id]) 
            X_c.index = range(X_c.shape[0])
            X_star_c.index = range(X_star_c.shape[0])

    if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
        region_res+=[corncob_cpg_local(f'{start}-{end}',meth[start:end,:].sum(axis=1),coverage[start:end,:].sum(axis=1),X,X_star,region)]

    cpg_res = pd.concat(cpg_res)
    if not region_res:
        region_res = None
    else:
        region_res = pd.concat(region_res)
        
    return cpg_res,region_res
    

@ray.remote
def corncob_cpg(cpg,meth_id,coverage_id,X_id,X_star_id,site_names_id):
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
        site_names_id (ray ID): ID of global value produced by ray.put(), pointing to a global array or list containing cpg names.
        
    Returns:
        pandas dataframe: Dataframe containing estimates of coeficents for the cpg. 
    """
    cc = Corncob_2(
                total=coverage_id[cpg,:],
                count=meth_id[cpg,:],
                X=X_id,
                X_star=X_star_id
            )
    
    e_m = cc.fit()
    res = cc.waltdt()[0]
    res["site"] = f'{site_names_id[cpg]}'
    return res
    
def corncob_cpg_local(site,meth,coverage,X,X_star,region=None,res=True,LL=False):
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
    
    e_m = cc.fit()
    if res:
        res = cc.waltdt()[0]
        if region != None:
            res["site"] = f'{region}_{site}'
        else:
            res["site"] = site
        if LL:
            return res,-e_m.fun
        else: 
            return res
    else:
        return -e_m.fun

class Corncob_2():
    """
        A python class to perform beta binomical regression on a single cpg.
        Based on pycorncob (https://github.com/jgolob/pycorncob/tree/main), with adjustments for DNA methylation data.
        Covariates or X should at least include an intercept column = 1
        total and count should be a an array / series with one entry per observation
        These must be in the same length and orientation as covariates
        
    """
    def __init__(self, total, count, X=None, X_star=None, phi_init=0.5):
        # Assertions here TODO
        
        self.total = total
        self.count = count
        self.X = X
        self.X_star = X_star
        self.n_param_abd = len(X.columns)
        self.n_param_disp = len(X_star.columns)
        self.n_ppar = len(X.columns) + len(X_star.columns)
        self.df_model = self.n_ppar
        self.df_residual = len(X) - self.df_model
        
        if (self.df_residual) < 0:
            raise ValueError("Model overspecified. Trying to fit more parameters than sample size.")
        
        self.param_names_abd = X.columns
        self.param_names_disp = X_star.columns
        self.phi_init = phi_init
        
        # Inits
        self.start_params = None
        # Final params set to none until fit
        self.theta = None
        self.params_abd = None
        self.params_disp = None
        self.converged = None
        self.method = None
        self.n_iter = None
        self.LogLike = None
        self.execution_time = None
        self.fit_out = None

    def corncob_init(self):
        m = sm.GLM(
            endog=pd.DataFrame([
                self.count, # Success
                self.total - self.count, # Failures
            ]).T,
            exog=self.X,
            family=sm.families.Binomial()
        ).fit()        
        return(
            list(m.params) + ([logit(self.phi_init)] * self.n_param_disp)
        )

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
        mu = expit(mu_wlink)
        phi = expit(phi_wlink) 
    
        a = mu*(1-phi)/phi
        b = (1-mu)*(1-phi)/phi
        LL = stats.betabinom.logpmf(
                k=self.count,
                n=self.total,
                a=a,
                b=b
            )
    
        return -1*np.sum(LL)  # Negative for minimization

    # Numerical Hessian calculation
    def compute_hessian(self):
        """
        Compute the Hessian matrix numerically using finite differences.
    
        Returns:
            numpy array: Hessian matrix.
        """
        W = self.count
        M = self.total        
        mu_wlink = np.matmul(
                self.X,
                self.theta[:self.n_param_abd]
            )
        phi_wlink = np.matmul(
                self.X_star,
                self.theta[-1*self.n_param_disp:]
            )
        mu = expit(mu_wlink)
        phi = expit(phi_wlink) 
        
        # define gam
        gam  = phi/(1 - phi)
        # Hold digammas
        dg1 = digamma(1/gam)
        dg2 = digamma(M + 1/gam)
        dg3 = digamma(M - (mu + W * gam - 1)/gam)
        dg4 = digamma((1 - mu)/gam)
        dg5 = digamma(mu/gam)
        dg6 = digamma(mu/gam + W)        

        # Hold partials - this part is fully generalized
        dldmu = (-dg3 + dg4 - dg5 + dg6)/gam
        dldgam = (-dg1 + dg2 + (mu - 1) * (dg3 - dg4) + mu * (dg5 - dg6))/(gam*gam)
        
        tg1 = polygamma(1, M - (mu + W * gam - 1)/gam)
        tg2 = polygamma(1, (1 - mu)/gam)
        tg3 = polygamma(1, mu/gam)
        tg4 = polygamma(1, mu/gam + W)
        tg5 = polygamma(1, 1/gam)
        tg6 = polygamma(1, M + 1/gam)
        dldmu2 = (tg1 - tg2 - tg3 + tg4)/np.power(gam, 2)
        dldgam2 = (2 * gam * dg1 + tg5 - 2 * gam * dg2 - tg6 + np.power(mu - 1, 2) * tg1 - 2 * gam * (mu - 1) * dg3 - np.power(mu, 2) * tg3 + np.power(mu, 2) * tg4 + np.power(mu - 1, 2) * (-1*tg2) + 2 * gam * (mu - 1) * dg4 - 2 * gam * mu * dg5 + 2 * gam * mu * dg6)/np.power(gam, 4)

        dldmdg = (gam * (dg3 - dg4 + dg5 - dg6) + (mu - 1) * (tg2 - tg1) + mu * (tg3 - tg4))/np.power(gam, 3)
        dpdb = self.X.T * (mu * (1 - mu) )
        dgdb = self.X_star.T * gam

        mid4 = dldmu * mu * (1 - mu) * (1 - 2 * mu)
        mid5 = dldgam * gam
        term4 = np.dot(
            self.X.T * mid4,
            self.X
        )
        term5 = np.dot(
            self.X_star.T * mid5,
            self.X_star
        )
        term1 = np.dot(
            dpdb,
            ((-1*dldmu2) * dpdb).T
        )
        term2 = np.dot(
            dpdb,
            ((-1*dldmdg) * dgdb).T
        )    
        term3 = np.dot(
            dgdb,
            ((-1*dldgam2) * dgdb).T
        )
        # Quadrants of hessian
        u_L = term1 - term4
        b_L = term2.T
        u_R = term2
        b_R = term3-term5
        return np.bmat([
            [u_L, u_R],
            [b_L, b_R]
        ])
    
    def waltdt(self):
        if self.theta is None:
            raise ValueError("No fitted parameters. Please run fit first")
        # Implicit else we have some parameters
        # Dataframes
        result_table_abd = pd.DataFrame(
            index=self.param_names_abd,
            columns=[
                'Estimate',
                'se',
                't',
                'p'
            ]
        )
        result_table_abd['Estimate'] = self.params_abd
        result_table_disp = pd.DataFrame(
            index=self.param_names_disp,
            columns=[
                'Estimate',
                'se',
                't',
                'p'
            ]
        )
        result_table_disp['Estimate'] = self.params_disp
        
        # Calculate SE
        try:
            # Compute the Hessian numerically at the optimized parameters
            hessian = self.compute_hessian()
            # Invert the Hessian to get the variance-covariance matrix
            covMat = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Hessian matrix is singular or ill-conditioned. Cannot compute variance-covariance matrix.")
            covMat = np.full_like(hessian, np.nan)  # Fill with NaNs

        # Implicit else we could calculate se
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            se = np.sqrt(np.diag(covMat))
        # Tvalues
        tvalue = self.theta/se
        # And pvalues
        pval = stats.t.sf(np.abs(tvalue), self.df_residual)*2

        
        result_table_abd['se'] = se[:self.n_param_abd]
        result_table_abd['t'] = tvalue[:self.n_param_abd]
        result_table_abd['p'] = pval[:self.n_param_abd]  

        result_table_disp['se'] = se[-1*self.n_param_disp:]
        result_table_disp['t'] = tvalue[-1*self.n_param_disp:]
        result_table_disp['p'] = pval[-1*self.n_param_disp:]
        
        return((
            result_table_abd,
            result_table_disp
        ))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method='Nelder-Mead', **kwds):
        if start_params == None:
            # Reasonable starting values
            start_params = self.corncob_init()
        
        # Fit the model using Nelder-Mead method and scipy minimize
        minimize_res = minimize(
            self.beta_binomial_log_likelihood,
            start_params,
            method=method
        )
        self.fit_out = minimize_res
        self.theta = minimize_res.x
        self.params_abd = minimize_res.x[:self.n_param_abd]
        self.params_disp = minimize_res.x[-1*self.n_param_disp:]
        self.converged = minimize_res.success
        self.method = method
        self.LogLike = -1*minimize_res.fun
        
        return minimize_res