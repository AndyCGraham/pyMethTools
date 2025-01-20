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
import ray
import warnings
from pyMethTools import bbll

class Corncob_2():
    """
        A python class to perform beta binomical regression on a single cpg.
        Based on pycorncob (https://github.com/jgolob/pycorncob/tree/main), with adjustments for DNA methylation data.
        Covariates or X should at least include an intercept column = 1
        total and count should be a an array / series with one entry per observation
        These must be in the same length and orientation as covariates
        
    """
    def __init__(self, total, count, X=None, X_star=None, phi_init=0.5,param_names_abd=["intercept"],param_names_disp=["intercept"],link="arcsin"):
        # Assertions here TODO

        self.meth = total
        self.coverage = count
        self.total = total[~np.isnan(total)]
        self.count = count[~np.isnan(total)]
        self.X = X[~np.isnan(total)]
        self.X_star = X_star[~np.isnan(total)]
        self.n_param_abd = X.shape[1]
        self.n_param_disp = X_star.shape[1]
        self.n_ppar = self.n_param_abd + self.n_param_disp
        self.df_model = self.n_ppar
        self.df_residual = self.X.shape[0] - self.df_model
        self.link = link
        
        if (self.df_residual) < 0:
            raise ValueError("Model overspecified. Trying to fit more parameters than sample size.")
        
        self.param_names_abd = param_names_abd
        self.param_names_disp = param_names_disp
        self.phi_init = phi_init
        
        # Inits
        self.start_params = []
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

    class ArcsineLink(Link):
        def __init__(self):
            self.__doc__ = "Arcsine link function"
        
        def _clean(self, p):
            """Ensure probabilities are within (0, 1)"""
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return p
    
        def inverse(self, z):
            """Inverse of the link function"""
            return np.sin(z)**2
    
        def inverse_deriv(self, z):
            """Derivative of the inverse link function"""
            return 2 * np.sin(z) * np.cos(z)
    
        def __call__(self, p):
            """Apply the link function"""
            p = self._clean(p)
            return np.arcsin(np.sqrt(p))

    def corncob_init(self):
        if self.link == "arcsin":
            link_function = self.ArcsineLink()
            m = sm.GLM(
                endog=pd.DataFrame([
                    self.count, # Success
                    self.total - self.count, # Failures
                ]).T,
                exog=self.X,
                family=sm.families.Binomial(link=link_function)
            ).fit()       
        else: 
            m = sm.GLM(
                endog=pd.DataFrame([
                    self.count, # Success
                    self.total - self.count, # Failures
                ]).T,
                exog=self.X,
                family=sm.families.Binomial(link=link_function)
            ).fit() 
        return(
            list(m.params) + ([np.arcsin(self.phi_init)] * self.n_param_disp)
        )

    @staticmethod
    def objective(beta, X, X_star, count, total, n_param_abd, n_param_disp):
        return bbll.bbll(beta, X, X_star, count, total, n_param_abd, n_param_disp)

    # Define the log-likelihood for beta-binomial regression (simplified version)
    def beta_binomial_log_likelihood(self, beta):
        """
        Compute the negative log-likelihood for beta-binomial regression.
    
        Parameters:
            beta (numpy array): Coefficients to estimate (p-dimensional vector).
            
        Returns:
            float: Negative log-likelihood.
        """
        # Compute linear predictors
        mu_wlink = self.X @ beta[:self.n_param_abd]
        phi_wlink = self.X_star @ beta[-self.n_param_disp:]
    
        # Transform to scale (0, 1)
        if self.link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif self.link == "logit":
            mu = expit(mu_wlink)
            phi = expit(phi_wlink)
    
        # Precompute shared terms
        phi_inv = (1 - phi) / phi
        a = mu * phi_inv
        b = (1 - mu) * phi_inv
    
        # Compute log-likelihood in a vectorized manner
        LL = stats.betabinom.logpmf(self.count, self.total, a, b)
        return -np.sum(LL)

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
        # Transform to scale (0, 1)
        if self.link == "arcsin":
            mu = np.sin(mu_wlink) ** 2
            phi = np.sin(phi_wlink) ** 2
        elif self.link == "logit":
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
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                # Compute the Hessian numerically at the optimized parameters
                hessian = self.compute_hessian()
                # Invert the Hessian to get the variance-covariance matrix
                covMat = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # print("Hessian matrix is singular or ill-conditioned. Cannot compute variance-covariance matrix.")
            covMat = np.full_like(hessian, np.nan)  # Fill with NaNs

        # Implicit else we could calculate se
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
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
    
    def fit(self, start_params=[], maxiter=10000, maxfev=5000, method='Nelder-Mead', **kwds):
        if len(start_params) == 0:
            # Reasonable starting values
            try:
                warnings.simplefilter('ignore', PerfectSeparationWarning)
                start_params = self.corncob_init()
            except:
                start_params = np.hstack([np.repeat(np.arcsin(self.phi_init),self.n_param_abd), np.repeat(np.arcsin(self.phi_init), self.n_param_disp)])
        
        # Fit the model using Nelder-Mead method and scipy minimize
        minimize_res = minimize(
            self.beta_binomial_log_likelihood,
            start_params,
            method=method,
            options={"maxiter": maxiter, "maxfev":maxfev}
        )
        self.fit_out = minimize_res
        self.theta = minimize_res.x
        self.params_abd = minimize_res.x[:self.n_param_abd]
        self.params_disp = minimize_res.x[-1*self.n_param_disp:]
        self.converged = minimize_res.success
        self.method = method
        self.LogLike = -1*minimize_res.fun
        
        return minimize_res

def fit_cpg_local(meth,coverage,X,X_star,maxiter=500,maxfev=500,start_params=[],param_names_abd=["intercept"],param_names_disp=["intercept"],link="arcsin"):
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
                    X_star=X_star,
                    param_names_abd=param_names_abd,
                    param_names_disp=param_names_disp,
                    link=link
                )
        
        e_m = cc.fit(maxiter=maxiter,maxfev=maxfev,start_params=start_params)
        return cc

def corncob_cpg_local(site,meth,coverage,X,X_star,maxiter=500,maxfev=500,region=None,res=True,LL=False,param_names_abd=["intercept"],param_names_disp=["intercept"],start_params=[],link="arcsin"):
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
                X_star=X_star,
                param_names_abd=param_names_abd,
                param_names_disp=param_names_disp,
                link=link
            )
    
    e_m = cc.fit(maxiter=maxiter,maxfev=maxfev,start_params=start_params)
    if res:
        res = cc.waltdt()[0].to_numpy()
            
        if region != None:
            site_range=site.split("-")
            res = np.hstack([res, np.tile(f'{region}',(res.shape[0],1)), np.tile(f'{region}_{site}',(res.shape[0],1)), 
                             np.tile(int(site_range[0]),(res.shape[0],1)), np.tile(int(site_range[1]),(res.shape[0],1))])
        if LL:
            return res,-e_m.fun
        else: 
            return res
    else:
        return -e_m.fun

def corncob_region_local(region,fits,min_cpgs=3,param_names_abd=["intercept"],param_names_disp=["intercept"],maxiter=500,maxfev=500):
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
        region_res=[]
        cpg_res = [fits[start].waltdt()[0]] #Add individual cpg res to cpg result table
        X_c = np.concatenate([fits[start].X,fits[start].X]) 
        X_star_c = np.concatenate([fits[start].X_star,fits[start].X_star]) 

        n_params_abd=fits[start].X.shape[1]
        n_params_disp=fits[start].X_star.shape[1]
        n_samples = fits[start].X.shape[0]
        n_params=n_params_abd+n_params_disp

        ll_s = fits[start].LogLike 
        
        while end < len(fits)+1:

            cpg_res += [fits[end-1].waltdt()[0]] #Add individual cpg res to cpg result table
            
            if start+min_cpgs < len(fits)+1: #Can we form a codistrib region
                
                cc_theta=np.vstack([fits[cpg].theta for cpg in range(start,end)]).mean(axis=0) #Estimate of joint parameters
                ll_c = corncob_cpg_local("",np.hstack([fit.count for fit in fits[start:end]]),
                                         np.hstack([fit.total for fit in fits[start:end]]),X_c,X_star_c,maxiter=maxiter,maxfev=maxfev,region=region,LL=True,res=False,
                                     param_names_abd=param_names_abd,param_names_disp=param_names_disp,start_params=cc_theta) 
                ll_s = ll_s + fits[end-1].LogLike
            
                bic_c = -2 * ll_c + (n_params) * np.log(n_samples) 
                bic_s = -2 * ll_s + (n_params*2) * np.log(n_samples) 
                
                end += 1
    
                #If sites come from the same distribution, keep extending the region
                if bic_c < bic_s:
                    ll_s=ll_c
                    X_c = np.concatenate([X_c,fits[start].X]) 
                    X_star_c = np.concatenate([X_star_c,fits[start].X_star]) 

                else: # Else start a new region
                    if (end-start) > min_cpgs+1: #Save region if number of cpgs > min_cpgs
                        cc_theta=np.vstack([fits[cpg].theta for cpg in range(start,end-2)]).mean(axis=0) #Estimate of joint parameters
                        region_res+=[corncob_cpg_local(f'{start}-{end-3}',np.vstack([fit.count for fit in fits[start:end-2]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end-2]]).sum(axis=0),
                                                       fits[start].X,fits[start].X_star,maxiter=maxiter,maxfev=maxfev,region=region,param_names_abd=param_names_abd,param_names_disp=param_names_disp,start_params=cc_theta)] #Add regions results to results table
                    start=end-2
                    X_c = np.concatenate([fits[start].X,fits[start].X]) 
                    X_star_c = np.concatenate([fits[start].X_star,fits[start].X_star]) 
                    ll_s = fits[start].LogLike 

            else:
                end += 1
    
        if (end-start) > min_cpgs+1: #Save final region if number of cpgs > min_cpgs
            region_res+=[corncob_cpg_local(f'{start}-{end-2}',np.vstack([fit.count for fit in fits[start:end]]).sum(axis=0),np.vstack([fit.total for fit in fits[start:end]]).sum(axis=0),
                                           X=fits[start].X,X_star=fits[start].X_star,maxiter=maxiter,maxfev=maxfev,region=region,param_names_abd=param_names_abd,param_names_disp=param_names_disp,start_params=cc_theta)]
    
        cpg_res = pd.concat(cpg_res)
        if not region_res:
            region_res = None
        else:
            region_res = pd.concat(region_res)
            
        return cpg_res,region_res

def beta_binomial_log_likelihood(count,total,X,X_star,beta,n_param_abd,n_param_disp,link="arcsin"):
        """
        Compute the negative log-likelihood for beta-binomial regression.
    
        Parameters:
            beta (numpy array): Coefficients to estimate (p-dimensional vector).
            
        Returns:
            float: Negative log-likelihood.
        """
        # Compute linear predictors
        mu_wlink = X @ beta[:n_param_abd]
        phi_wlink = X_star @ beta[-n_param_disp:]
    
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
    
        # Compute log-likelihood in a vectorized manner
        LL = stats.betabinom.logpmf(count, total, a, b)
        return np.sum(LL) 