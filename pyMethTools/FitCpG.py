from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from statsmodels.genmod.families.links import Link
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import betabinom
from scipy.optimize import minimize,OptimizeResult
from scipy import linalg
from scipy.stats import t,rankdata,norm
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit,expit, digamma, polygamma
from numpy.linalg import svd, inv, LinAlgError
import ray
import warnings

class FitCpG():
    """
        A python class to perform beta binomical regression on a single cpg.
        Based on pycorncob (https://github.com/jgolob/pycorncob/tree/main), with adjustments for DNA methylation data.
        Covariates or X should at least include an intercept column = 1
        total and count should be a an array / series with one entry per observation
        These must be in the same length and orientation as covariates
        
    """
    def __init__(self, total, count, X = None, X_star = None, phi_init = 0.5,link = "arcsin",
                 fit_method = "gls",sample_weights = None,theta = None,c0 = 0.1):
        # Assertions here TODO
        
        if sample_weights is not None:
            self.sample_weights = sample_weights[~np.isnan(total)]
        else:
            self.sample_weights = np.ones(sum(~np.isnan(total)))
        self.total = total[~np.isnan(total)]
        self.count = count[~np.isnan(count)]
        self.Z = np.arcsin(2*((self.count + c0)/(self.total+2*c0))-1) # Transformed methylation proportions
        self.n_samples = self.total.shape[0]
        self.X = X[~np.isnan(total)]
        self.X_star = X_star[~np.isnan(total)]
        self.n_param_abd = X.shape[1]
        self.n_param_disp = X_star.shape[1]
        self.n_ppar = self.n_param_abd + self.n_param_disp
        self.df_model = self.n_ppar
        self.df_residual = self.X.shape[0] - self.df_model
        self.link = link
        self.fit_method = fit_method
        
        if (self.df_residual) < 0:
            raise ValueError("Model overspecified. Trying to fit more parameters than sample size.")
        
        self.phi_init = phi_init
        
        # Inits
        self.start_params = []
        self.theta = theta
        if theta is not None:
            self.params_abd = theta[:self.n_param_abd]
            self.params_disp = theta[-self.n_param_disp:]
        else:
            self.params_abd = None
            self.params_disp = None
        
        # Final params set to none until fit
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
        
        def deriv(self, p):
            """Derivative of the link function: g'(p) = 1/(2*sqrt(p*(1-p)))"""
            p = self._clean(p)
            return 1.0/(2.0*np.sqrt(p*(1-p)))
    
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
                family=sm.families.Binomial(link=link_function,check_link=False)
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
        
        dispersion_est = np.sum(m.resid_pearson**2) / m.df_resid
        return(
            list(m.params) + ([np.arcsin(np.sqrt(dispersion_est))] * self.n_param_disp)
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
        Adjusted for arcsin link function.
        """
        W = self.total
        M = self.count        
        
        mu_wlink = np.matmul(self.X, self.theta[:self.n_param_abd])
        phi = np.matmul(self.X_star, self.theta[-self.n_param_disp:])
        
        # Transform to (0,1) scale using arcsin link
        if self.link == "arcsin":
            mu = ((np.sin(mu_wlink) + 1) / 2) 
            dmu_deta = np.sin(2 * mu_wlink)  # First derivative of arcsin link
            d2mu_deta2 = 2 * np.cos(2 * mu_wlink)  # Second derivative of arcsin link
        elif self.link == "logit":
            mu = expit(mu_wlink)
            dmu_deta = mu * (1 - mu)  # Standard logit derivative
            d2mu_deta2 = dmu_deta * (1 - 2 * mu)
    
        # Define gamma (dispersion transformation)
        gam = phi / (1 - phi)
    
        # Digamma function values
        dg1 = digamma(1 / gam)
        dg2 = digamma(M + 1 / gam)
        dg3 = digamma(M - (mu + W * gam - 1) / gam)
        dg4 = digamma((1 - mu) / gam)
        dg5 = digamma(mu / gam)
        dg6 = digamma(mu / gam + W)
    
        # First derivatives
        dldmu = (-dg3 + dg4 - dg5 + dg6) / gam
        dldgam = (-dg1 + dg2 + (mu - 1) * (dg3 - dg4) + mu * (dg5 - dg6)) / (gam ** 2)
    
        # Second derivatives
        tg1 = polygamma(1, M - (mu + W * gam - 1) / gam)
        tg2 = polygamma(1, (1 - mu) / gam)
        tg3 = polygamma(1, mu / gam)
        tg4 = polygamma(1, mu / gam + W)
        tg5 = polygamma(1, 1 / gam)
        tg6 = polygamma(1, M + 1 / gam)
    
        dldmu2 = (tg1 - tg2 - tg3 + tg4) / (gam ** 2)
        dldgam2 = (2 * gam * dg1 + tg5 - 2 * gam * dg2 - tg6 +
                   ((mu - 1) ** 2) * tg1 - 2 * gam * (mu - 1) * dg3 -
                   (mu ** 2) * tg3 + (mu ** 2) * tg4 + ((mu - 1) ** 2) * (-tg2) +
                   2 * gam * (mu - 1) * dg4 - 2 * gam * mu * dg5 + 2 * gam * mu * dg6) / (gam ** 4)
    
        dldmdg = (gam * (dg3 - dg4 + dg5 - dg6) + (mu - 1) * (tg2 - tg1) + mu * (tg3 - tg4)) / (gam ** 3)
    
        # Partial derivatives with link function applied
        dpdb = self.X.T * dmu_deta
        dgdb = self.X_star.T * gam
    
        # Hessian components
        term4 = np.dot(self.X.T * (dldmu * d2mu_deta2), self.X)
        term5 = np.dot(self.X_star.T * gam, self.X_star)
    
        term1 = np.dot(dpdb, ((-dldmu2) * dpdb).T)
        term2 = np.dot(dpdb, ((-dldmdg) * dgdb).T)
        term3 = np.dot(dgdb, ((-dldgam2) * dgdb).T)
    
        # Hessian quadrants
        u_L = term1 - term4
        b_L = term2.T
        u_R = term2
        b_R = term3 - term5

        # Comput hessian from quadrants
        upper_row = np.concatenate((u_L, u_R), axis=1)
        lower_row = np.concatenate((b_L, b_R), axis=1)
        Hessian = np.concatenate((upper_row, lower_row), axis=0)
    
        return Hessian
    
    def gls(self, c1 = 0.001):
        """
        Fits the model using a two-stage weighted least squares procedure.
        """
        # Subset non-missing samples where total > 0.
        ix = self.total > 0
        if np.mean(ix) < 1:
            X = self.X[ix, :]
            count = self.count[ix]
            total = self.total[ix]
            Z = self.Z[ix]
            sample_weights = self.sample_weights[ix] if self.sample_weights is not None else np.ones_like(total)
            if X.shape[0] < X.shape[1] + 1:
                res = OptimizeResult()
                res.x = np.repeat(np.nan, X.shape[1] + 1)
                res.se_beta0 = np.nan
                res.var_beta0 = np.nan
                res.success = False
                res.message = "Not enough degrees of freedom. Require more samples than parameters."
                return res
            # Check full rank.
            U, s, Vt = np.linalg.svd(X)
            if np.any(np.abs(s) < 1e-10):
                res = OptimizeResult()
                res.x = np.repeat(np.nan, X.shape[1] + 1)
                res.se_beta0 = np.nan
                res.var_beta0 = np.nan
                res.success = False
                res.message = "Design matrix is not full rank"
                return res
        else:
            X = self.X
            count = self.count
            total = self.total
            Z = self.Z
            sample_weights = self.sample_weights if self.sample_weights is not None else np.ones_like(total)

        # Update sample count and parameter count.
        n_samples = X.shape[0]
        p = self.n_param_abd

        # --- First round of weighted least squares ---
        combined_weights = total * sample_weights
        XTVinv = (X * combined_weights[:, np.newaxis]).T
        XtX = np.dot(XTVinv, X)
        try:
            beta0 = np.linalg.solve(XtX, np.dot(XTVinv, Z))
        except np.linalg.LinAlgError:
            res = OptimizeResult()
            res.x = np.repeat(np.nan, X.shape[1] + 1)
            res.se_beta0 = np.nan
            res.var_beta0 = np.nan
            res.success = False
            res.message = "Unable to solve for beta0 (first round)."
            return res

        # --- Estimate dispersion (phiHat) ---
        residual = Z - np.dot(X, beta0)
        weighted_sum_sq = np.sum((residual**2) * combined_weights)
        effective_df = np.sum(sample_weights) - p
        phiHat = (weighted_sum_sq - effective_df) * np.sum(sample_weights) / (effective_df * np.sum((total - 1) * sample_weights))
        phiHat = np.clip(phiHat, c1, 1 - c1)

        # --- Second round regression with updated weights ---
        # Define new weights based on phiHat.
        Vinv = sample_weights * total / (1 + (total - 1) * phiHat)
        XTVinv = (X * Vinv[:, np.newaxis]).T
        XtX = np.dot(XTVinv, X)
        try:
            # Solve for beta0 without explicit inversion.
            beta0 = np.linalg.solve(XtX, np.dot(XTVinv, Z))
        except np.linalg.LinAlgError:
            res = OptimizeResult()
            res.x = np.repeat(np.nan, X.shape[1] + 1)
            res.se_beta0 = np.nan
            res.var_beta0 = np.nan
            res.success = False
            res.message = "Unable to solve for beta0 (second round)."
            return res

        # Compute standard errors without explicit inversion.
        I = np.eye(XtX.shape[0])
        XtX_inv = np.linalg.solve(XtX, I)  # This gives the full inverse of XtX.
        se_beta0 = np.sqrt(np.diag(XtX_inv))
        var_beta0 = XtX_inv.flatten()

        res = OptimizeResult()
        res.x = np.append(beta0, phiHat)  # Coefficients and dispersion in one array.
        res.se_beta0 = se_beta0
        res.var_beta0 = var_beta0
        res.success = True
        res.message = "gls completed successfully."
        res.var = var_beta0
        return res
        
    def fit(self, 
            start_params = [], 
            maxiter = 10000, 
            maxfev = 5000, 
            minimize_method = 'Nelder-Mead',
            **kwds):
        # Fit the model using Nelder-Mead method and scipy minimize
        if self.fit_method=="gls":
            minimize_res = self.gls()
        if minimize_res.success == False or self.fit_method=="mle":
            if self.fit_method=="gls": 
                print("Failed GLS")

            if len(start_params) == 0:
                # Reasonable starting values
                try:
                    warnings.simplefilter('ignore', PerfectSeparationWarning)
                    start_params = self.corncob_init()
                except:
                    start_params = np.hstack([np.repeat(np.arcsin(self.phi_init),self.n_param_abd), np.repeat(np.arcsin(self.phi_init), self.n_param_disp)])

            #Minimize the log-likelihood
            minimize_res = minimize(
                self.beta_binomial_log_likelihood,
                start_params,
                method=minimize_method,
                options={"maxiter": maxiter, "maxfev":maxfev}
            )
            self.LogLike = -1*minimize_res.fun
            # Compute the Hessian numerically at the optimized parameters
            hessian = self.compute_hessian()
            # Invert the Hessian to get the variance-covariance matrix
            covMat = np.linalg.inv(hessian) @ hessian @ np.linalg.inv(hessian)
            minimize_res.se_beta0 = np.sqrt(np.diag(covMat))

        return minimize_res