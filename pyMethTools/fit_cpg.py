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
    
    def fit(self, start_params=None, maxiter=10000, maxfev=5000, method='Nelder-Mead', **kwds):
        if start_params == None:
            # Reasonable starting values
            try:
                start_params = self.corncob_init()
            except:
                start_params = np.repeat(logit(self.phi_init),self.n_param_abd) + ([logit(self.phi_init)] * self.n_param_disp)
        
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
            res["region"] = f'{region}'
            site=site.split("-")
            res["range"] = [range(int(site[0]),int(site[1])+1) for _ in range(res.shape[0])]
        else:
            res["site"] = site
        if LL:
            return res,-e_m.fun
        else: 
            return res
    else:
        return -e_m.fun