import numpy as np
from scipy.linalg import solve, inv
from scipy.optimize import OptimizeResult

class FitRegion:
    def __init__(self, X, count, total, cpg_idx, sample_idx, sample_weights=None,
                 link = "arcsin", fit_method = "gls", theta = None, c0 = 0.1):
        """
        X:         (n_obs × p) fixed‑effect design
        count:     (n_obs,) methylated counts
        total:     (n_obs,) total reads
        cpg_idx:   (n_obs,) in 0..m-1 specifying which cpg each observation belongs to
        sample_idx:(n_obs,) in 0..J-1 specifying which sample each observation belongs to
        """
        self.total = total[~np.isnan(count)]
        self.count = count[~np.isnan(total)]
        self.X = X
        self.Z = np.arcsin(2*((self.count + c0)/(self.total+2*c0))-1) # Transformed methylation proportions
        self.cpg, self.samp = cpg_idx, sample_idx
        self.n_obs, self.p     = X.shape
        self.m                 = np.unique(cpg_idx).size
        self.J                 = np.unique(sample_idx).size
        if sample_weights is not None:
            self.sample_weights = sample_weights[~np.isnan(total)]
        else:
            self.sample_weights = np.ones(sum(~np.isnan(total)))

        self.n_samples = self.total.shape[0]
        self.X = X[~np.isnan(total)]
        self.n_param_abd = X.shape[1]
        self.n_param_disp = 1
        self.n_ppar = self.n_param_abd + self.n_param_disp
        self.df_model = self.n_ppar
        self.df_residual = self.X.shape[0] - self.df_model
        self.link = link
        self.fit_method = fit_method
        
        if (self.df_residual) < 0:
            raise ValueError("Model overspecified. Trying to fit more parameters than sample size.")
        
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

    def fit(self, tol = 1e-6, max_iter = 50):
        # 1) initial GLS fit
        res = self.gls()
        β, φ = res.x[:-1], res.x[-1]

        # initialize Σ = I_m
        Σ = np.eye(self.m)

        for iteration in range(max_iter):
            # --- E‑step: BLUP for each sample j ---
            b = np.zeros((self.J, self.m))
            for j in range(self.J):
                # indices for sample j
                idx = np.where(self.samp == j)[0]
                Xj, Zlj = self.X[idx], self.Z[idx]
                cj, tj = self.count[idx], self.total[idx]
                # residual vector r_j = Zl_j - X_j β
                rj = Zlj - Xj.dot(β)
                # working‐variance R_j diag
                Vj = tj / (1 + (tj - 1)*φ)  # BB variance on logit approx
                Rj = np.diag(Vj)
                # BLUP: b_j = Σ (Σ + R_j)⁻¹ r_j
                M = Σ + Rj
                bj = Σ.dot(solve(M, rj))
                b[j] = bj

            # --- M‑step: update Σ as empirical cov of b ---
            Σ_new = (b.T.dot(b)) / self.J

            # --- Re-fit fixed effects with GLS on Zl - b_j ---
            # form adjusted response
            Zl_adj = np.zeros_like(self.Z)
            for j in range(self.J):
                idx = np.where(self.samp == j)[0]
                Zl_adj[idx] = self.Z[idx] - b[j, self.cpg[idx]]
            # temporarily replace self.Z for gls
            old_Z = getattr(self, 'Zl', None)
            self.Z = Zl_adj
            res = self.gls()
            β_new, φ_new = res.x[:-1], res.x[-1]
            se_beta = res.se_beta0
            # restore if needed
            if old_Z is not None: self.Z = old_Z

            # --- check convergence ---
            if (np.allclose(β, β_new, atol=tol) and
                abs(φ-φ_new)<tol      and
                np.allclose(Σ, Σ_new, atol=tol)):
                β, φ, Σ = β_new, φ_new, Σ_new
                break

            β, φ, Σ = β_new, φ_new, Σ_new

        # pack results
        out = {
            'beta':β, 'se':se_beta, 'phi':φ, 'Sigma':Σ, 'random_effects':b,
            'iterations':iteration+1
        }
        return out
