import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg.lapack as lapack
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from dataclasses import dataclass
from functools import partial
from .meta_optimizer import MetaOptimizer
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from enum import Enum
from .utils import read_init, ProjectData, logger_print, openers
import dill

class GLSRefinement(str, Enum):
    none = 'none'
    fixed = 'fixed'
    full = 'full'

class GOFStat(str, Enum):
    fov = 'fov'
    corr = 'corr'

class GOFStatMode(str, Enum):
    residual = 'residual'
    total = 'total'
    
class FOVMeanMode(str, Enum):
    null = 'null'
    gls = 'gls'
    knn = 'knn'

def null_space_transform(Q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute V^T Y where V is the orthogonal complement to Q, using Householder 
    transformations via LAPACK's dormqr. 
    
    Parameters:
    Q (ndarray): p x r semi-orthogonal matrix where Q^T Q = I_r, r <= p. 
    Y (ndarray): p x n matrix. 
    
    Returns:
    VT_Y (ndarray): (p - r) x n matrix representing V^T Y (float64).
    """

    Y = np.array(Y, order='F', copy=True)

    p, r = Q.shape

    if r > p:
        raise ValueError(f"Number of columns r ({r}) cannot exceed number of rows p ({p}) in Q.")
        
    # 1. Compute QR factorization of Q
    # Need a copy of Q because 'raw' QR might modify it slightly in some versions/backends,
    # even though documentation often says it doesn't. Using overwrite_a=True below is safer.
    Q_copy = np.array(Q, order='F', dtype=np.float64) # Fortran order often preferred by LAPACK
    qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
    if info_qr != 0:
        raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
    # qr_a now contains R in upper triangle and reflectors below diagonal (overwritten Q_copy)
    
    # 2. Prepare matrix Z (to be modified by dormqr)

    # 3. Apply Q_full^T to Z using dormqr
    lwork = -1
    # Use Z's shape here for the query, pass dummy Z
    _, work_query, _ = lapack.dormqr('L', 'T', qr_a, tau, np.empty_like(Y), lwork=lwork, overwrite_c=True)
    optimal_lwork = int(work_query[0].real)
    lwork = max(1, optimal_lwork)


    # Actual application
    q_mult_y, work_actual, info_ormqr = lapack.dormqr('L', 'T', qr_a, tau, Y, 
                                                      lwork=lwork, overwrite_c=True)
    
    if info_ormqr != 0:
        print("--- Debug Info Before dormqr Failure ---")
        print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
        print(f"qr_a shape: {qr_a.shape}, dtype: {qr_a.dtype}, order: {'F' if qr_a.flags.f_contiguous else 'C'}")
        print(f"tau shape: {tau.shape}, dtype: {tau.dtype}")
        print(f"Y shape: {Y.shape}, dtype: {Y.dtype}, order: {'F' if Y.flags.f_contiguous else 'C'}")
        print(f"lwork: {lwork}")
        print("--- End Debug Info ---")
        raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")

    VT_Y = q_mult_y[r:, :] 
    return VT_Y
@dataclass(frozen=True)
class LowrankDecomposition:
    Q: np.ndarray
    S: np.ndarray
    V: np.ndarray

    def null_space_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute V^T Y where V is the orthogonal complement to Q, using Householder 
        transformations via LAPACK's dormqr. 
        
        Parameters:
        Q (ndarray): p x r semi-orthogonal matrix where Q^T Q = I_r, r <= p. 
        Y (ndarray): p x n matrix. 
        
        Returns:
        VT_Y (ndarray): (p - r) x n matrix representing V^T Y (float64).
        """
        Y = np.array(Y, order='F', copy=True)
        Q = np.array(self.Q).astype(np.float64, copy=False)

        p, r = Q.shape

        if r > p:
            raise ValueError(f"Number of columns r ({r}) cannot exceed number of rows p ({p}) in Q.")
            
        # 1. Compute QR factorization of Q
        # Need a copy of Q because 'raw' QR might modify it slightly in some versions/backends,
        # even though documentation often says it doesn't. Using overwrite_a=True below is safer.
        Q_copy = np.array(Q, order='F', dtype=np.float64) # Fortran order often preferred by LAPACK
        qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
        if info_qr != 0:
            raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
        # qr_a now contains R in upper triangle and reflectors below diagonal (overwritten Q_copy)
        
        # 2. Prepare matrix Z (to be modified by dormqr)

        # 3. Apply Q_full^T to Z using dormqr
        lwork = -1
        # Use Z's shape here for the query, pass dummy Z
        _, work_query, _ = lapack.dormqr('L', 'T', qr_a, tau, np.empty_like(Y), lwork=lwork, overwrite_c=True)
        optimal_lwork = int(work_query[0].real)
        lwork = max(1, optimal_lwork)


        # Actual application
        q_mult_y, work_actual, info_ormqr = lapack.dormqr('L', 'T', qr_a, tau, Y, 
                                                          lwork=lwork, overwrite_c=True)
        
        if info_ormqr != 0:
            print("--- Debug Info Before dormqr Failure ---")
            print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
            print(f"qr_a shape: {qr_a.shape}, dtype: {qr_a.dtype}, order: {'F' if qr_a.flags.f_contiguous else 'C'}")
            print(f"tau shape: {tau.shape}, dtype: {tau.dtype}")
            print(f"Y shape: {Y.shape}, dtype: {Y.dtype}, order: {'F' if Y.flags.f_contiguous else 'C'}")
            print(f"lwork: {lwork}")
            print("--- End Debug Info ---")
            raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")

        VT_Y = q_mult_y[r:, :] 
        return VT_Y
    
    
    def adjoint_null_space_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes V @ Z, the inverse of the null_space_transform.
        V is the orthogonal complement to Q. Z is typically the output of
        null_space_transform (i.e., Z = V^T Y).
        
        Parameters:
        Q (ndarray): The p x r semi-orthogonal matrix from the class instance.
        Z (ndarray): (p - r) x n matrix. 
        
        Returns:
        V_Z (ndarray): p x n matrix representing V @ Z (float64).
        """
        Q = np.array(self.Q).astype(np.float64, copy=False)
        p, r = Q.shape
        
        if Z.ndim != 2:
            raise ValueError("Input Z must be a 2D array.")
            
        if Z.shape[0] != p - r:
            raise ValueError(f"Input Z must have p-r = {p-r} rows, but got {Z.shape[0]}.")
        
        n = Z.shape[1]
        
        # Ensure Z is in a compatible format for LAPACK
        Z = np.array(Z, dtype=np.float64, copy=False)
    
        # 2. Compute the QR factorization of Q to get Householder reflectors
        # This part is identical to the forward transform.
        Q_copy = np.array(Q, order='F', dtype=np.float64)
        qr_a, tau, work_qr, info_qr = lapack.dgeqrf(Q_copy, overwrite_a=True)
        if info_qr != 0:
            raise RuntimeError(f"LAPACK dgeqrf failed with info = {info_qr}")
            
        # 3. Create the padded matrix [0; Z]
        # This is the key step for the inverse transformation.
        Y_padded = np.vstack([np.zeros((r, n)), Z])
        Y_padded = np.array(Y_padded, order='F', copy=True)
    
        # 4. Apply Q_full (no transpose) to the padded matrix using dormqr
        # Note: transr='N' for no transpose
        
        # Workspace query
        lwork = -1
        _, work_query, _ = lapack.dormqr('L', 'N', qr_a, tau, Y_padded, lwork=lwork, overwrite_c=True)
        optimal_lwork = int(work_query[0].real)
        lwork = max(1, optimal_lwork)
    
        # Actual application of the transformation
        V_Z, work_actual, info_ormqr = lapack.dormqr('L', 'N', qr_a, tau, Y_padded, 
                                                      lwork=lwork, overwrite_c=True)
        
        if info_ormqr != 0:
            raise RuntimeError(f"LAPACK dormqr failed with info = {info_ormqr}")
    
        return V_Z



@dataclass
class TransformedData:
    Y: np.ndarray
    B: np.ndarray
    group_inds: list
    group_inds_inv: np.ndarray
    K : np.ndarray = None
    
@dataclass(frozen=True)
class ErrorVarianceEstimates:
    variance: np.ndarray
    promotor: np.ndarray
    fim: np.ndarray
    loglik: float
    loglik_start: float

@dataclass(frozen=True)
class MotifVarianceEstimates:
    motif: np.ndarray
    group: np.ndarray
    fim: np.ndarray
    fixed_group: int
    loglik: float
    loglik_start: float

@dataclass(frozen=True)
class MotifMeanEstimates:
    mean: np.ndarray
    fim: np.ndarray

@dataclass(frozen=True)
class PromoterMeanEstimates:
    mean: np.ndarray

@dataclass(frozen=True)
class SampleMeanEstimates:
    mean: np.ndarray

@dataclass(frozen=True)
class FitResult:
    error_variance: ErrorVarianceEstimates
    motif_variance: MotifVarianceEstimates 
    motif_mean: MotifMeanEstimates 
    promoter_mean: PromoterMeanEstimates
    sample_mean: SampleMeanEstimates
    group_names: list
    clustering: np.ndarray = None
    clustered_B: np.ndarray = None
    promoter_inds_to_drop: list = None
    
def ones_nullspace(n: int):
    res = np.zeros((n - 1, n), dtype=float)
    for i in range(1, n):
        norm = (1 / i + 1) ** 0.5
        res[i - 1, :i] = -1 / i / norm
        res[i - 1, i] = 1 / norm
    return res

def ones_nullspace_transform(x):
    n, m = x.shape
    if n <= 1:
        return np.zeros((0, m), dtype=x.dtype)

    Y = np.zeros((n - 1, m), dtype=float) 
    current_sum = x[0, :].astype(float)

    for r in range(n - 1): 
        i = r + 1 
        sqrt_i_i_plus_1 = np.sqrt(i * (i + 1)) 
        
        # Coefficients for row r of Y (which uses row i-1 = r of H)
        coeff1 = -1.0 / sqrt_i_i_plus_1
        coeff2 = np.sqrt(i / (i + 1))
        Y[r, :] = coeff1 * current_sum + coeff2 * x[r + 1, :]

        # Update current_sum for the next iteration (to become sum_{k=0}^{r+1} X[k,:])
        if r < n - 2: # Avoid adding beyond X's bounds on the last iteration
             current_sum += x[r + 1, :]
    return Y

def ones_nullspace_transform_transpose(X: np.ndarray) -> np.ndarray:
    n, m = X.shape
    n = n + 1 

    if n == 1:
         output_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else float
         return np.zeros((1, m), dtype=output_dtype)

    output_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else float
    Y = np.zeros((n, m), dtype=output_dtype)

    current_suffix_sum = np.zeros(m, dtype=output_dtype)

    for k in range(n - 2, -1, -1):
        i = k + 1.0 

        sqrt_term_i_ip1 = np.sqrt(i * (i + 1.0))
        coeff_pos = i / sqrt_term_i_ip1
        coeff_neg = -1.0 / sqrt_term_i_ip1


        Y[k + 1, :] = coeff_pos * X[k, :] + current_suffix_sum

        current_suffix_sum += coeff_neg * X[k, :]

    Y[0, :] = current_suffix_sum

    return Y

def lowrank_decomposition(X: np.ndarray, rel_eps=1e-15) -> LowrankDecomposition:
    svd = jnp.linalg.svd
    q, s, v = [np.array(t) for t in svd(X, full_matrices=False)]
    if rel_eps is not None:
        max_sv = max(s)
        n = len(s)
        for r in range(n):
            if s[r] / max_sv < rel_eps:
                r -= 1
                break
        r += 1
        s = s[:r]
        q = q[:, :r]
        v = v[:r]
    return LowrankDecomposition(q, s, v)

def transform_data(data, std_y=False, std_b=False, helmert=True, weights=None) -> TransformedData:
    try:
        B = data.B_orig
        Y = data.Y_orig
        group_inds = data.original_inds
    except:
        B = data.B
        Y = data.Y
        group_inds = data.group_inds
    if weights is not None:
        B = B * weights.reshape(-1, 1)
        Y = Y * weights.reshape(-1, 1)
        if weights.std()  == 0:
            weights = None
    if std_b:
        B /= B.std(axis=0, keepdims=True)
    if helmert:
        if weights is None:
            # F_p = ones_nullspace(len(Y))
            # Y = F_p @ Y
            # B = F_p @ B
            Y = ones_nullspace_transform(Y)
            B = ones_nullspace_transform(B)
        else:
            weights = weights.reshape(-1, 1)
            weights /= np.linalg.norm(weights)
            Y = null_space_transform(weights, Y)
            B = null_space_transform(weights, B)
    group_inds_inv = list()
    d = dict()
    for i, items in enumerate(group_inds):
        for j in items:
            d[j] = i
    for i in sorted(d.keys()):
        group_inds_inv.append(d[i])
    group_inds_inv = np.array(group_inds_inv)
    return TransformedData(Y=Y, B=B, 
                           group_inds=group_inds,
                           group_inds_inv=group_inds_inv)

def loglik_error(d: jnp.ndarray, Qn_Y: jnp.ndarray, group_inds_inv: jnp.ndarray) -> float:
    d = d.at[group_inds_inv].get()
    logdet_D = jnp.log(d).sum()
    d = 1 / d
    logdet_FDF = logdet_D + jnp.log(d.mean())
    K = d * d.sum()
    xi = jnp.exp(logdet_D - logdet_FDF - jnp.log(len(d)))
    Y1 = Qn_Y * K
    Y2 = Qn_Y @ d.reshape(-1, 1)
    m = len(Qn_Y)
    return xi * (jnp.einsum('ij,ij->', Y1, Qn_Y) - (Y2.T @ Y2).flatten()[0]) + m * logdet_FDF

def loglik_error_full(x: jnp.ndarray, Y: jnp.ndarray, Q_C: jnp.ndarray, group_inds_inv: jnp.ndarray, D_fix_val: float, D_fix_ind: int) -> float:
    p, s = Y.shape
    r = Q_C.shape[1]
    x = x ** 2
    D = x.at[:-p].get() + 1e-4
    if D_fix_ind is not None:
        D = jnp.insert(D, D_fix_ind, D_fix_val)
    S = x.at[-p:].get() + 1e-3
    S = 1 / S
    D = 1 / D
    D = D.at[group_inds_inv].get()
    w_D = D.sum()
    M = Q_C.T * S @ Q_C
    YD = Y * D
    YS = Y.T * S
    YD = YD - jnp.outer(YD.sum(axis=-1), D / w_D)
    YS = YS - YS @ Q_C @ jnp.linalg.inv(M) @ Q_C.T * S
    vec = jnp.einsum('ij,ji->', YD, YS)
    logdet_D = (p-r) * (-jnp.log(D).sum()  + jnp.log(w_D))
    logdet_S = (s-1) * (jnp.linalg.slogdet(M)[0] - jnp.log(S).sum())
    logdet = logdet_D + logdet_S
    return vec + logdet

# def loglik_error_full(x: jnp.ndarray, Y: jnp.ndarray, Q_C: jnp.ndarray, group_inds_inv: jnp.ndarray, D_fix_val: float, D_fix_ind: int) -> float:
#     p, s = Y.shape
#     x = x ** 2
#     D = x.at[:-p].get()
#     if D_fix_ind is not None:
#         D = jnp.insert(D, D_fix_ind, D_fix_val)
#     S = x.at[-p:].get()
#     S = 1 / S
#     D = 1 / D
#     D = D.at[group_inds_inv].get()
#     w_D = D.sum()
#     M = (Q_C.T * S) @ Q_C

#     YD = Y * D
#     YS = Y.T * S
#     YD = YD - jnp.outer(YD.sum(axis=-1), D / w_D)
#     YS = YS - YS @ Q_C @ jnp.linalg.inv(M) @ Q_C.T
#     vec = jnp.einsum('ij,ji->', YD, YS)
#     logdet_D = -(p-1) * (jnp.log(D).sum()  - w_D)
#     logdet_S = (s-1) * (jnp.linalg.slogdet(M)[0] - jnp.log(S).sum())
#     logdet = logdet_D + logdet_S
#     return vec + logdet
    
    
def loglik_error_grad(d: jnp.ndarray, Qn_Y: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray) -> jnp.ndarray:
    d = d.at[group_inds_inv].get()
    logdet_D = jnp.log(d).sum()
    d = 1 / d
    logdet_FDF = logdet_D + jnp.log(d.mean())
    K = d * d.sum()
    xi = jnp.exp(logdet_D - logdet_FDF - jnp.log(len(d)))
    Y1 = Qn_Y * K
    Y2 = Qn_Y @ d.reshape(-1, 1) 
    Z = Y1 - Y2 @ d.reshape(1, -1)
    g = len(Qn_Y) * xi * (K - d ** 2) - xi ** 2 * jnp.einsum('ji,ji->i', Z, Z)
    return jnp.array([g[ind].sum() for ind in group_inds])

def loglik_motifs(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                  D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                  G_fix_ind=None, G_fix_val=1.0, drop_sigma=False, _motif_zero=None) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    if _motif_zero is not None:
        Sigma = Sigma.at[_motif_zero].set(0)
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    cov = jnp.kron(D_A, D_B) + 1
    logdet = jnp.log(cov).sum()
    v = (Q_B.T * Sigma @ Z * G @ Q_A).flatten('F')
    loglik = -(v ** 2 / cov).sum() + logdet
    return loglik 

def loglik_motifs_joint(x: jnp.ndarray, Z: jnp.ndarray, Sigma_hat: jnp.ndarray,
                        group_inds_inv: jnp.ndarray, g: int):
    p, s = Z.shape
    
    # --- Setup S ---
    x_sq = x ** 2
    D_diag = x_sq[:g][group_inds_inv]
    G_diag = x_sq[g:][group_inds_inv]
    
    # S is diagonal, so we only work with the (ps,)-shaped diagonal vector
    S_diag = jnp.kron(G_diag, Sigma_hat) + jnp.kron(D_diag, jnp.ones_like(Sigma_hat))
    S_inv_diag = 1.0 / S_diag

    # --- Log-Determinant Term ---
    # log(det(S)) + log(det(D_N))
    # D_N = (1/s) * sum(S_i^-1), so log(det(D_N)) = p*log(1/s) + log(det(sum(S_i^-1)))
    # We ignore the constant p*log(1/s)
    logdet_S = jnp.log(S_diag).sum()
    
    # Reshape S_inv_diag to (p, s) to easily sum the blocks
    S_inv_reshaped = S_inv_diag.reshape(p, s, order='F')
    U_diag = S_inv_reshaped.sum(axis=1)  # U = sum(S_i^-1), shape (p,)
    
    logdet_DN = jnp.log(U_diag).sum()
    logdet_term = logdet_S + logdet_DN

    # --- Quadratic Term ---
    # vec(Z)^T S^-1 vec(Z) - (v^T U^-1 v), where v = sum(S_i^-1 z_i)

    # Term 1: vec(Z)^T S^-1 vec(Z)
    # This is an element-wise product summed over all ps elements.
    term1 = (Z**2 * S_inv_reshaped).sum()

    # Term 2 (Correction): -v^T U^-1 v
    # v = sum_{i=1 to s} S_i^{-1} z_i
    v_vec = (S_inv_reshaped * Z).sum(axis=1) # Shape (p,)
    
    # U is diagonal with diagonal elements U_diag. U^-1 is diagonal with 1/U_diag.
    # v^T U^-1 v is a sum over p elements.
    term2 = -(v_vec**2 / U_diag).sum()

    quadratic_term = term1 + term2
    
    # --- Final Log-Likelihood ---
    # The formula for a Gaussian log-likelihood has a negative sign.
    # loglik = -0.5 * (quadratic_term + logdet_term)
    # If you are minimizing the negative log-likelihood, you would return 0.5 * (...)
    return 0.5 * (quadratic_term + logdet_term)
    
    

def loglik_motifs_naive(x: jnp.ndarray, Y_Fn: jnp.ndarray, B, BTB, FDF, Fn,
                        group_inds_inv, fix_group, fix_val):
    x = jnp.array(x)
    Sigma = x.at[:len(BTB)].get() 
    G = x.at[len(BTB):].get() 
    if fix_group is not None:
        G = jnp.insert(G, fix_group, fix_val)
    G = G.at[group_inds_inv].get()
    mx = jnp.kron(Fn * G @ Fn.T, B * Sigma @ B.T)
    mx = mx + jnp.kron(FDF, jnp.identity(len(B)))
    Y_Fn = Y_Fn.reshape((-1, 1), order='F')
    return (Y_Fn.T @ jnp.linalg.inv(mx) @ Y_Fn).flatten()[0] + jnp.linalg.slogdet(mx)[1]

def loglik_motifs_grad(x: jnp.ndarray, Z: jnp.ndarray, BTB: jnp.ndarray,
                  D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                  group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                  drop_sigma=False,
                  _motif_zero=None) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    if _motif_zero is not None:
        Sigma = Sigma.at[_motif_zero].set(0)
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    s = 1 / (jnp.kron(D_A, D_B) + 1)
    M = s.reshape((len(Q_B), Q_A.shape[1]), order='F')
    Lambda_base = (Q_B.T * Sigma @ Z * G @ Q_A) * M
    
    # Derivative w.r.t. to tau/Sigma
    Lambda = Q_B @ Lambda_base
    vec_term = jnp.einsum('ij,ij->i', Lambda, Lambda) / (Sigma ** 2)
    Z = (Q_B.T * Sigma @ BTB) * Q_B.T
    det_term = (s.reshape(1, -1) @ jnp.kron(D_A.reshape(-1,1), Z)).flatten() / Sigma
    grad_tau = -vec_term + det_term
    # Derivative w.r.t to nu/G
    Lambda = Lambda_base @ Q_A.T
    vec_term = jnp.einsum('ji,ji->i', Lambda, Lambda) / (G ** 2)
    Z = (Q_A.T * G @ D_product_inv) * Q_A.T
    det_term = (s.reshape(1, -1) @ jnp.kron(Z, D_B.reshape(-1,1))).flatten() / G
    grad_nu = -vec_term + det_term
    grad_nu = jnp.array([grad_nu.at[inds].get().sum() for inds in group_inds])
    if G_fix_ind is not None:
        grad_nu = jnp.delete(grad_nu, G_fix_ind)
    grad = jnp.append(grad_tau, grad_nu)
    if _motif_zero is not None:
        grad = grad.at[_motif_zero].set(0)
    if drop_sigma:
        grad = grad[len(BTB):]
    return grad 

def loglik_motifs_fim_naive(x: jnp.ndarray, B: jnp.ndarray, 
                      D: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0):
    def cov(x):
        Sigma = x.at[:B.shape[1]].get() 
        G = x.at[B.shape[1]:].get()
        if G_fix_ind is not None:
            G = jnp.insert(G, G_fix_ind, G_fix_val)
        G = G.at[group_inds_inv].get()
        tD = D.at[group_inds_inv].get()
        H = ones_nullspace(len(tD))
        L = jnp.linalg.cholesky(H * tD @ H.T)
        L = jnp.linalg.inv(L)
        A = L @ H * G @ H.T @ L.T
        C = B * Sigma @ B.T
        S_hat = jnp.kron(A, C)
        S_hat = S_hat + np.identity(len(S_hat))
        return S_hat
    S_hat = cov(x)
    S_hat = np.linalg.inv(S_hat)
    grad = jax.jacrev(cov)(x)
    fim = np.zeros((len(x), len(x)), dtype=float)
    vec = S_hat @ grad
    for i in range(len(x)):
        for j in range(i, len(x)):
            fim[i, j] = jnp.trace(vec[..., i] @ vec[..., j]) / 2
            fim[j, i] = fim[i, j]
    return fim
        

def loglik_motifs_fim(x: jnp.ndarray, BTB: jnp.ndarray, 
                      D_product_inv: jnp.ndarray, group_inds_inv: jnp.ndarray,
                      group_inds: jnp.ndarray, G_fix_ind=None, G_fix_val=1.0,
                      drop_sigma=False) -> float:
    if drop_sigma:
        x = jnp.append(jnp.ones(len(BTB)), x)
    Sigma = x.at[:len(BTB)].get() ** 0.5
    G = x.at[len(BTB):].get()
    if G_fix_ind is not None:
        G = jnp.insert(G, G_fix_ind, G_fix_val)
    G = G ** 0.5
    G = G.at[group_inds_inv].get()
    D_A, Q_A = jnp.linalg.eigh(G.reshape(-1, 1) * D_product_inv * G)
    D_B, Q_B = jnp.linalg.eigh(Sigma.reshape(-1, 1) * BTB * Sigma)
    D_A = jnp.where(D_A > 0, D_A, 0.0)
    D_B = jnp.where(D_B > 0, D_B, 0.0)
    s = 1 / (jnp.kron(D_A, D_B) + 1)
    indices = jnp.arange(len(s), dtype=int)

    indices = len(G) * (indices % len(Sigma)) + indices // len(Sigma)
    s_permuted = s.at[indices].get()
    BTCQ = BTB * Sigma @ Q_B
    D_prod_Q = D_product_inv * G @ Q_A
   
    group_loadings = np.zeros((len(G), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)
    indices = jnp.arange(0, len(s), dtype=int).reshape((len(G), len(Sigma)))
    
    @jax.jit
    def f_tau(k, mx):
        ind = indices.at[k].get()
        S_k = s.at[ind].get()
        Lambda_k = BTCQ * S_k @ Q_B.T
        return mx + D_A.at[k].get() ** 2 * Lambda_k * Lambda_k.T

    FIM_tau = jnp.zeros((len(Sigma), len(Sigma)), dtype=float)
    FIM_tau = jax.lax.fori_loop(0, len(D_A), f_tau, FIM_tau) / 2
    FIM_tau = FIM_tau * jnp.outer(1 / Sigma, 1 / Sigma)
    @jax.jit
    def f_nu(k, mx):
        ind = indices.at[k].get()
        S_k = s_permuted.at[ind].get()
        Gamma_k = D_prod_Q * S_k @ Q_A.T
        return mx + D_B.at[k].get() ** 2 * Gamma_k * Gamma_k.T
    
    indices = indices.reshape(indices.shape[::-1])
    FIM_nu = jnp.zeros((len(G), len(G)), dtype=float)
    FIM_nu = jax.lax.fori_loop(0, len(D_B), f_nu, FIM_nu) / 2
    FIM_nu = FIM_nu * jnp.outer(1 / G, 1 / G)
    FIM_nu = group_loadings.T @ FIM_nu @ group_loadings
    indices = jnp.arange(0, len(s), dtype=int)
    indices_mod = indices % len(Sigma)
    indices_div = indices // len(Sigma)
    indices = jnp.array(list(np.ndindex((len(Sigma), len(G)))))
    zeta = s ** 2 * D_A.at[indices_div].get() * D_B.at[indices_mod].get()
    Psi = BTCQ * Q_B
    K = D_prod_Q
    Theta = K * Q_A 

    @jax.jit
    def f_tau_nu(ind):
        i, j = ind
        psi_i = Psi.at[i, indices_mod].get()
        theta_j = Theta.at[j, indices_div].get()
        return (zeta * psi_i * theta_j).sum()
    
    FIM_tau_nu = jnp.zeros((len(Sigma), len(G)), dtype=float)
    FIM_tau_nu = FIM_tau_nu.at[*indices.T].set(jax.lax.map(f_tau_nu, indices, batch_size=32))
    FIM_tau_nu = FIM_tau_nu * jnp.outer(1 / Sigma, 1 / G) / 2
    FIM_tau_nu = FIM_tau_nu @ group_loadings
    
    if G_fix_ind is not None:
        FIM_nu = jnp.delete(jnp.delete(FIM_nu, G_fix_ind, axis=0), G_fix_ind, axis=1)
        FIM_tau_nu = jnp.delete(FIM_tau_nu, G_fix_ind, axis=1)
    if drop_sigma:
        FIM_tau = jnp.identity(FIM_tau.shape[0])
        FIM_tau_nu = jnp.zeros_like(FIM_tau_nu)
    FIM = jnp.block([[FIM_tau, FIM_tau_nu],
                     [FIM_tau_nu.T, FIM_nu]])
    return FIM


def calc_error_variance_fim(data: TransformedData, error_variance: jnp.ndarray):
    d = 1 / jnp.array(error_variance).at[data.group_inds_inv].get()
    d = d / d.sum() ** 0.5
    D_product_inv = jnp.outer(-d, d)
    D_product_inv = jnp.fill_diagonal(D_product_inv,
                                      D_product_inv.diagonal() + d * d.sum(),
                                      inplace=False )
    fim = D_product_inv * D_product_inv.T / 2
    group_inds = data.group_inds
    group_loadings = np.zeros((len(d), len(group_inds)), dtype=int)
    for i, indices in enumerate(group_inds):
        group_loadings[indices, i] = 1
    group_loadings = jnp.array(group_loadings)
    return group_loadings.T @ fim @ group_loadings

def estimate_error_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                            verbose=False) -> ErrorVarianceEstimates:
    p = B_decomposition.Q.shape[0]
    Y = B_decomposition.null_space_transform(data.Y)
    d0 = jnp.array([np.var(Y[:, inds]) for inds in data.group_inds])

    fun = partial(loglik_error, Qn_Y=Y, group_inds_inv=data.group_inds_inv)
    grad = partial(loglik_error_grad, Qn_Y=Y, group_inds_inv=data.group_inds_inv,
                   group_inds=data.group_inds)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad,  num_steps_momentum=15, 
                        )
    res = opt.optimize(d0)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    
    fim = calc_error_variance_fim(data, res.x)
    return ErrorVarianceEstimates(np.array(res.x), np.ones(p),
                                  np.array(fim),
                                  loglik_start=res.start_loglik,
                                  loglik=res.fun)


def estimate_error_variance_full(data: TransformedData,
                                 B_decomposition: LowrankDecomposition,
                                 error_variance: ErrorVarianceEstimates,
                                 original_data: TransformedData = None,
                                 verbose=False) -> ErrorVarianceEstimates:
    # Y = B_decomposition.null_space_transform(data.Y)
    Y = data.Y
    d0 = error_variance.variance 
    D_fix_ind = np.argmin(d0)
    D_fix_val = d0[D_fix_ind]
    d0 = np.delete(d0, D_fix_ind)
    fun = partial(loglik_error_full, Y=Y, Q_C=B_decomposition.Q, group_inds_inv=data.group_inds_inv, D_fix_val=D_fix_val, D_fix_ind=D_fix_ind)
    fun = jax.jit(jax.value_and_grad(fun, argnums=0))
    from scipy.optimize import minimize
    if original_data is not None:
        Y0 = original_data.Y
        Y0 = Y0 - Y0.mean(axis=0, keepdims=True) - Y0.mean(axis=1, keepdims=True) + Y0.mean()
        D = error_variance.variance
        D = D[original_data.group_inds_inv]
        Y0 = Y0 / D ** (-0.5)
        prom_x0 = Y0.var(axis=1)
    else:
        prom_x0 = jnp.ones(len(data.Y))
    x0 = jnp.append(d0, prom_x0) ** 0.5
    res = minimize(fun, x0, jac=True,
                   method='TNC'#'L-BFGS-B', 
                   # options={'maxiter': 10000},
                   )
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    x = res.x ** 2
    D = x[:-len(data.Y)] + 1e-4
    D = np.insert(D, D_fix_ind, D_fix_val)
    S = x[-len(data.Y):] + 1e-3
    fim = error_variance.fim # TODO
    
    return ErrorVarianceEstimates(np.array(D), np.array(S), np.array(fim),
                                  loglik_start=error_variance.loglik_start,
                                  loglik=res.fun)

def estimate_promoter_mean(data: TransformedData,
                            B_decomposition: LowrankDecomposition,
                            error_variance: ErrorVarianceEstimates,
                            verbose=False) -> PromoterMeanEstimates:
    
    D = error_variance.variance[data.group_inds_inv]
    Y = jnp.array(data.Y)
    Q_C = jnp.array(B_decomposition.Q)
    w = (1 / D).sum()
    mean = Y @ (1 / D.reshape(-1, 1))
    mean = mean - Q_C @ (Q_C.T @ mean)
    weights = error_variance.promotor ** -0.5
    if np.std(weights) > 1e-12:
        q = weights / np.linalg.norm(weights)
        

        decomp_null = LowrankDecomposition(
            Q=q.reshape(-1, 1), 
            S=np.array([]), 
            V=np.array([])
        )
        

        mean_2d = mean.reshape(-1, 1)
        mean_2d = decomp_null.adjoint_null_space_transform(mean_2d)
        mean = mean_2d.flatten()
    else:

        mean = ones_nullspace_transform_transpose(mean)
    mean = mean / w
    return PromoterMeanEstimates(mean)

def _estimate_motif_variance_mom(Y, B, ind_fix, fix_value, eps=1e-14):
    # Gamma = (B^T B)^-1
    BTB = B.T @ B
    try:
        Gamma = np.linalg.inv(BTB)
    except np.linalg.LinAlgError:
        raise ValueError("B must have full column rank to be invertible.")
        
    # W = (B^T B)^-1 B^T
    W = Gamma @ B.T
    Z = W @ Y
    
    # E[Z_ij^2] = sigma_i^2 * g_j + Gamma_ii
    # We estimate sigma_i^2 * g_j by subtracting the known noise bias Gamma_ii.
    Gamma_diag = np.diag(Gamma)
    
    # S_ij = Z_ij^2 - Gamma_ii
    S = np.square(Z) - Gamma_diag[:, None]
    
    #  R_i approx sum(g) * sigma_i^2
    R = np.sum(S, axis=1)
    # C_j approx sum(sigma^2) * g_j
    C = np.sum(S, axis=0)
    
    # Using the ratio of column sums: g_j / g_fixed = C_j / C_fixed
    if C[ind_fix] == 0:
        scale_factor = 0
    else:
        scale_factor = fix_value / C[ind_fix]
        
    g_est = C * scale_factor
    
    # sigma_i^2 = R_i / sum(g)
    sum_g_est = np.sum(g_est)
    
    if sum_g_est == 0:
        sigma_sq_est = np.zeros_like(R)
    else:
        sigma_sq_est = R / sum_g_est
        
    return np.clip(sigma_sq_est, eps, float('inf')), np.clip(g_est, eps, float('inf'))

def estimate_motif_variance(data: TransformedData, B_decomposition: LowrankDecomposition,
                             error_variance: ErrorVarianceEstimates,
                             original_data: TransformedData = None,
                             verbose=False) -> MotifVarianceEstimates:
    multiplier = 1e-2
    D = jnp.array(error_variance.variance)
    j = jnp.argsort(D)[len(D) // 2]
    fix = D.at[j].get() * multiplier
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    d = 1 / D.at[data.group_inds_inv].get()
    
    if original_data is not None:
        Y = original_data.Y
        B = original_data.B
        Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
        Y = Y * d ** 0.5
        B = B - B.mean(axis=0, keepdims=True)
        Sigma0, G0 = _estimate_motif_variance_mom(Y, B, j, 1e-1)
        G0 = G0 / d 
        G0 = np.array([G0[inds].mean() for inds in original_data.group_inds])
        scaler = fix / G0[j]
        G0 = G0 * scaler
        Sigma0 = Sigma0 / scaler
        G0 = np.delete(G0, j)
    else:
        Sigma0, G0 = jnp.ones(len(BTB), dtype=float), np.repeat(fix, len(D) - 1)
    
    d = d / d.sum() ** 0.5
    D_product_inv = jnp.outer(-d, d)
    D_product_inv = jnp.fill_diagonal(D_product_inv,
                                      D_product_inv.diagonal() + d * d.sum(),
                                      inplace=False )
    Z = data.B.T @ data.Y @ D_product_inv

    x0 = jnp.append(Sigma0, G0)
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, G_fix_ind=j, G_fix_val=fix)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  G_fix_ind=j, G_fix_val=fix)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50)
    res = opt.optimize(x0)
    if not np.isfinite(res.fun):
        print(res)
        print(res.x)
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    Sigma = res.x[:len(BTB)]
    G = res.x[len(BTB):]
    
    G = jnp.insert(G, j, fix)
    fim = partial(loglik_motifs_fim, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  G_fix_ind=j, G_fix_val=fix)
    f = fim(res.x)
    eig = np.linalg.eigvalsh(f).min()
    print('FIM min eig', eig)
    if eig < 0:
        eig = list()
        epsilons =  [1e-23, 1e-18, 1e-15, 1e-12, 1e-9, 1e-8,
                     1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for eps in epsilons:
            x = res.x.copy()
            x = x.at[:len(BTB)].set(jnp.clip(x.at[:len(BTB)].get(), eps, float('inf')))
            f = fim(x)
            eig.append(np.linalg.eigvalsh(f).min())
            print(eps, eig[-1])
            if eig[-1] > 0:
                break
        i = np.argmax(eig)
        eps = epsilons[i]
        x = res.x.copy()
        x = x.at[:len(BTB)].set(jnp.clip(x.at[:len(BTB)].get(), eps, float('inf')))
        fim = fim(x)
    else:
        fim = f
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=j, loglik_start=res.start_loglik,
                                  loglik=res.fun)


def estimate_motif_variance_identity(data: TransformedData, B_decomposition: LowrankDecomposition,
                                     error_variance: ErrorVarianceEstimates,
                                     verbose=False) -> MotifVarianceEstimates:
    D = jnp.array(error_variance.variance)
    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    d = 1 / D.at[data.group_inds_inv].get()
    d = d / d.sum() ** 0.5
    D_product_inv = jnp.outer(-d, d)
    D_product_inv = jnp.fill_diagonal(D_product_inv,
                                      D_product_inv.diagonal() + d * d.sum(),
                                      inplace=False )
    # d = 1 / D.at[data.group_inds_inv].get()
    # D_product_inv_alt = jnp.diag(d) - jnp.outer(d, d) / d.sum()

    Z = data.B.T @ data.Y @ D_product_inv
    x0 = np.repeat(0.1, len(D))
    fun = partial(loglik_motifs, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, drop_sigma=True)
    grad = partial(loglik_motifs_grad, Z=Z, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  drop_sigma=True)
    fun = jax.jit(fun)
    grad = jax.jit(grad)
    opt = MetaOptimizer(fun, grad, num_steps_momentum=50)
    res = opt.optimize(x0)
    
    if verbose:
        print('-' * 15)
        print(res)
        print('-' * 15)
    Sigma = np.ones(len(BTB))
    G = res.x
    fim = partial(loglik_motifs_fim, BTB=BTB, D_product_inv=D_product_inv,
                  group_inds_inv=data.group_inds_inv, group_inds=data.group_inds,
                  drop_sigma=True)
    fim = fim(res.x)
    eig = jnp.linalg.eigh(fim)[0].min()
    print('FIM min eig', eig)
    return MotifVarianceEstimates(motif=np.array(Sigma), group=np.array(G), fim=np.array(fim),
                                  fixed_group=None, loglik_start=res.start_loglik,
                                  loglik=res.fun)

def estimate_motif_mean(data: TransformedData, B_decomposition: LowrankDecomposition,
                         error_variance: ErrorVarianceEstimates,
                         motif_variance: MotifVarianceEstimates,
                         promoter_mean: PromoterMeanEstimates) -> MotifMeanEstimates:
    D = jnp.array(error_variance.variance)
    Sigma = jnp.array(motif_variance.motif)
    G = jnp.array(motif_variance.group)
    mu_p = jnp.array(promoter_mean.mean)
    
    d = D ** 0.5
    d = d.at[data.group_inds_inv].get()
    g = G ** 0.5
    g = g.at[data.group_inds_inv].get()
    R = G ** 0.5 / D
    r = R.at[data.group_inds_inv].get()

    BTB = B_decomposition.V.T * B_decomposition.S ** 2 @ B_decomposition.V
    A = jnp.sqrt(Sigma).reshape(-1, 1) * BTB
    # Fp = ones_nullspace(len(data.Y) + 1)
    # Y_tilde = (data.Y - Fp @ mu_p.reshape(-1, 1)) / d
    weights = error_variance.promotor ** -0.5
    if np.std(weights) > 1e-12:
        # Weighted case: Use Householder transform matching the weights
        # Normalize weights to get the projection vector Q
        q = weights / np.linalg.norm(weights)
        # Apply the transform
        # null_space_transform expects Q as (p, r) and Y as (p, n)
        mu_p_transformed = null_space_transform(q.reshape(-1, 1), mu_p.reshape(-1, 1))
    else:
        # Standard case: Use Helmert transform
        mu_p_transformed = ones_nullspace_transform(mu_p.reshape(-1, 1))
    Y_tilde = (data.Y - mu_p_transformed) / d
    Y_hat = jnp.sqrt(Sigma).reshape(-1,1) *  data.B.T @ Y_tilde * g / d
    D_B, Q_B = jnp.linalg.eigh(jnp.sqrt(Sigma).reshape(-1, 1) * BTB * jnp.sqrt(Sigma))
    At_QB = A.T @ Q_B
    w = (1 / d ** 2).sum()
    hat_g = (d / g) ** 2
    S = 1 / (hat_g.reshape(-1,1) + D_B) / d.reshape(-1,1) ** 2
    S = S.T
    mx = w * BTB - At_QB * S.sum(axis=-1) @ At_QB.T
    b = data.B.T @ (Y_tilde @ (1 / d).reshape(-1, 1))
    b = b - At_QB @ ((S / r) * (Q_B.T @ Y_hat)).sum(axis=-1, keepdims=True) 
    fim = mx
    mx = jnp.linalg.pinv(mx)
    mu_m = mx @ b
    return MotifMeanEstimates(np.array(mu_m), np.array(fim))

def estimate_sample_mean(data: TransformedData, error_variance: ErrorVarianceEstimates, 
                         motif_variance: MotifVarianceEstimates, promoter_mean: PromoterMeanEstimates,
                         motif_mean: MotifMeanEstimates):
    Y = data.Y
    B = data.B
    Y = Y - promoter_mean.mean.reshape(-1, 1) - B @ motif_mean.mean.reshape(-1, 1)
    
    Y = jnp.asarray(Y)
    B = jnp.asarray(B)
    Sigma = jnp.asarray(motif_variance.motif)
    G = jnp.asarray(motif_variance.group)
    D = jnp.asarray(error_variance.variance)
    G = G.at[data.group_inds_inv].get()
    D = D.at[data.group_inds_inv].get()
    a_vec = (error_variance.promotor ** (-0.5))

    p, m = B.shape
    sqrt_Sigma = np.sqrt(Sigma).reshape(1, -1)
    C = B * sqrt_Sigma
    U, S, _ = jnp.linalg.svd(C, full_matrices=False)
    S_sq = S ** 2


    a = a_vec.T @ U
    sum_Y = a_vec.T @ Y
    a_sq_norm = np.sum(a_vec ** 2)


    UT_Y = U.T @ Y
    a_sq = a ** 2
    sum_a_sq = np.sum(a_sq)

    a_UT_Y = a[:, np.newaxis] * UT_Y


    num_part1 = np.sum(a_UT_Y / (G[np.newaxis, :] * S_sq[:, np.newaxis] + D[np.newaxis, :]), axis=0)
    num_part2 = (sum_Y - np.sum(a_UT_Y, axis=0)) / D
    numerator = num_part1 + num_part2


    denom_part1 = np.sum(a_sq[:, np.newaxis] / (G[np.newaxis, :] * S_sq[:, np.newaxis] + D[np.newaxis, :]), axis=0)
    denom_part2 = (a_sq_norm - sum_a_sq) / D
    denominator = denom_part1 + denom_part2

    mu = numerator / denominator

    return SampleMeanEstimates(np.array(mu).reshape(-1, 1))

@dataclass(frozen=True)
class ActivitiesPrediction:
    U: np.ndarray
    U_raw: np.ndarray
    filtered_motifs: np.ndarray
    tau_groups: dict
    clustering: tuple[np.ndarray, np.ndarray] = None
    _cov: tuple[np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray] = None
    
    def cov(self) -> np.ndarray:
        assert self._cov is not None
        Q_hat, S, sigma, nu, n, tau_mult = self._cov
        for sigma, nu, n, tau_mult in zip(sigma, nu, n, tau_mult):
            tau = nu / sigma * tau_mult
            D = n * S + 1 / tau
            D = 1 / D * sigma
            D = D ** 0.5
            Q_hat2 = Q_hat * D
            c = np.array(Q_hat2 @ Q_hat2.T, dtype=float)
            yield c

def predict_activities(data: TransformedData, fit: FitResult,
                       filter_motifs=True, filter_order=5,
                       tau_search=True, tau_left=0.1,  tau_right=1.0, tau_num=15,
                       clustering_search=False, k_min=0.1, k_max=0.9, k_num=6, 
                       cv_repeats=3, cv_splits=5,
                       pinv=False) -> ActivitiesPrediction:

    # def _sol(BT_Y_sum, BT_B, Sigma, sigma, nu, n: int, tau_mult=1.0):
    def _sol(BT_Y_sum, Q_hat, S, sigma, nu, n: int, tau_mult=1.0, BT_B=None):
        tau = nu / sigma * tau_mult
        if pinv:
            tau_mult = np.clip(tau_mult - 1, 0.0, a_max=float('inf'))
            sol = jnp.linalg.pinv(BT_B + tau_mult * jnp.identity(len(BT_B))) @ BT_Y_sum
        else:
            D = ( n * S + 1 / tau) ** (-0.5)
            Q_hat = Q_hat * D
            sol = Q_hat @ Q_hat.T @ BT_Y_sum
        return sol

    Sigma = fit.motif_variance.motif
    G = fit.motif_variance.group
    D = fit.error_variance.variance
    group_inds = data.group_inds
    a_vec = fit.error_variance.promotor ** -0.5
    mu_p = (a_vec * fit.promoter_mean.mean.flatten()).reshape(-1, 1)
    mu_m = fit.motif_mean.mean.reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    B = data.B
    Y = data.Y
    # Y = Y - Y.mean(axis=0, keepdims=True) - Y.mean(axis=1, keepdims=True) + Y.mean()
    # B = B - B.mean(axis=0, keepdims=True)
    Y = Y - mu_p - B @ mu_m - np.outer(a_vec, mu_s.flatten())
    # # 3. [THE FIX] Center B. Weighted column sums become 0.
    # # We calculate the weighted mean of each motif column
    w_norm_sq = np.dot(a_vec, a_vec)
    weighted_col_means = (a_vec @ B) / w_norm_sq
    
    # # Subtract it. Now B matches the geometry of Y.
    B0 = B
    B = B - np.outer(a_vec, weighted_col_means)
    # print(np.linalg.norm(B-B0))
    if filter_motifs:
        inds = np.log10(Sigma) >= (np.median(np.log10(Sigma)) - filter_order)
        B = B[:, inds]
        Sigma = Sigma[inds]
        mu_m = mu_m[inds]
        filtered_motifs = np.where(~inds)[0]
    else:
        filtered_motifs = list()
    clusters = defaultdict(list)
    if clustering_search:
        from tqdm import tqdm
        for n_cluster in tqdm([10, 25, 50, 75, 100, 150, 200, 500, B.shape[1]]):
            if n_cluster == B.shape[1]:
                Bc = B
                Sigma_c = Sigma
            else:
                Bc, c = cluster_data(B, mode=ClusteringMode.KMeans, num_clusters=n_cluster)
                Sigma_c = c * Sigma @ c.T 
                Sigma_c = Sigma_c.diagonal()
            rkf = RepeatedKFold(n_repeats=cv_repeats, random_state=1, n_splits=cv_splits)
            for train_inds, test_inds in rkf.split(Y):
                B_train = Bc[train_inds]
                B_test = Bc[test_inds]
                Y_train = Y[train_inds]
                Y_test = Y[test_inds]

                BT_Y = B_train.T @ Y_train
                if not pinv:
                    B_train = B_train * Sigma ** 0.5
                BT_B = B_train.T @ B_train
                S, Q_hat = jnp.linalg.eigh(BT_B)
                Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=1, BT_B=BT_B)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    clusters[n_cluster].append(diff)
        clusters = {n: np.mean(v) for n, v in clusters.items()}
    else:
        clusters = {B.shape[1]: 0} 
    best_clust = min(clusters, key=lambda x: clusters[x])
    if best_clust == B.shape[1]:
        clust = None
        pass
    else:
        B, clust = cluster_data(B, mode=ClusteringMode.KMeans, num_clusters=best_clust)
        Sigma = c * Sigma @ c.T
        Sigma = Sigma.diagonal()
    tau_groups = defaultdict(lambda: defaultdict(list))
    if tau_search:
        from tqdm import tqdm
        # stats = defaultdict(float)
        rkf = RepeatedKFold(n_repeats=cv_repeats, random_state=1, n_splits=cv_splits)
        for train_inds, test_inds in tqdm(list(rkf.split(Y))):
            B_train = B[train_inds]
            B_test = B[test_inds]
            Y_train = Y[train_inds]
            Y_test = Y[test_inds]
            BT_Y = B_train.T @ Y_train
            if not pinv:
                B_train = B_train * Sigma ** 0.5
            BT_B = B_train.T @ B_train
            S, Q_hat = jnp.linalg.eigh(BT_B)
            Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat
            for tau in np.linspace(tau_left, tau_right, num=tau_num):
                # pi = jnp.linalg.pinv(B)
                for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
                    # all_inds.extend(inds)
                    BT_Y_sub = BT_Y[:, inds]
                    U = _sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=tau, BT_B=BT_B)
                    diff = ((Y_test[:, inds] - B_test @ U[:, np.argsort(inds)]) ** 2).mean()
                    tau_groups[i][tau].append(diff)
        tau_groups = {g: min(v, key=lambda x: np.mean(v[x])) for g, v in tau_groups.items()}
    else:
        tau_groups = {i: 1.0 for i in range(len(group_inds))}
    BT_Y = B.T @ Y
    if not pinv:
        B = B * Sigma ** 0.5
    BT_B = B.T @ B
    S, Q_hat = jnp.linalg.eigh(BT_B)
    Q_hat = (Sigma ** 0.5).reshape(-1, 1) * Q_hat

    U = list()
    U0 = list()
    sizes = list()
    all_inds = list()
    tau_mults = list()
    for i, (inds, sigma, nu) in enumerate(zip(group_inds, D, G)):
        tau = tau_groups[i]
        tau_mults.append(tau)
        all_inds.extend(inds)
        BT_Y_sub = BT_Y[:, inds]
        sizes.append(len(inds))
        U_pred = _sol(BT_Y_sub.sum(axis=-1, keepdims=True), Q_hat, S,
                      sigma, nu, len(inds), tau_mult=tau,
                      BT_B=BT_B)
        U.append(U_pred)
        U0.append(_sol(BT_Y_sub, Q_hat, S, sigma, nu, 1, tau_mult=tau, BT_B=BT_B))
    U = np.concatenate(U, axis=-1)
    U0 = np.concatenate(U0, axis=-1)[:, np.argsort(all_inds)]
    return ActivitiesPrediction(U, U_raw=U0,
                                filtered_motifs=filtered_motifs,
                                tau_groups=tau_groups,
                                clustering=(B, clust) if clust is not None else None,
                                _cov=(Q_hat, S, D, G, sizes, tau_mults))
def solve_als(
    Z: np.ndarray,
    S: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    mu_0: np.ndarray,
    nu_0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    orthogonalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    p, s = Z.shape
    W = S.reshape((p, s), order='F')
    c = c.flatten()
    d = d.flatten()
    mu = mu_0.flatten().astype(float)
    nu = nu_0.flatten().astype(float)
    epsilon = 1e-14
    mu_denominators = np.sum(W * (c**2)[:, np.newaxis], axis=0) + epsilon
    nu_denominators = np.sum(W * (d**2), axis=1) + epsilon

    for i in range(max_iter):
        mu_old = mu.copy()
        nu_old = nu.copy()
        term1_mu = np.sum(W * c[:, np.newaxis] * Z, axis=0)
        sum_Wcn = np.sum(W * c[:, np.newaxis] * nu[:, np.newaxis], axis=0)
        term2_mu = d * sum_Wcn
        mu = (term1_mu - term2_mu) / mu_denominators
        term1_nu = np.sum(W * d * Z, axis=1)
        sum_Wdm = np.sum(W * d * mu, axis=1)
        term2_nu = c * sum_Wdm
        nu = (term1_nu - term2_nu) / nu_denominators
        mu_change = np.linalg.norm(mu - mu_old) / (np.linalg.norm(mu_old) + epsilon)
        nu_change = np.linalg.norm(nu - nu_old) / (np.linalg.norm(nu_old) + epsilon)
        if mu_change < tol and nu_change < tol:
            # Quieting the output for this run
            print(f"ALS converged successfully after {i+1} iterations.")
            break
    if orthogonalize:
        a = np.dot(mu, d) / (np.dot(d, d) + 1e-12)
        mu = mu - a * d
        nu = nu + a * c
    return mu[..., None], nu[..., None]

def null_space_transform_jax(Q: jax.Array, Y: jax.Array) -> jax.Array:
    p, r = Q.shape
    A = Q
    Y_transformed = Y

    def _householder_loop_body(j, state):
        A, Y_transformed = state
        col_j = A[:, j]
        mask = (jnp.arange(p) >= j)
        x_padded = col_j * mask
        sign_x_j = jnp.where(col_j[j] >= 0, 1.0, -1.0)
        alpha = -sign_x_j * jnp.linalg.norm(x_padded)
        v = x_padded.at[j].add(-alpha)
        v_norm_sq = jnp.dot(v, v)
        tau = jnp.where(v_norm_sq < 1e-24, 0.0, 2.0 / v_norm_sq)

        def update_A_func(A_in):
            w_A = jnp.dot(v, A_in)
            update_A = tau * jnp.outer(v, w_A)
            col_mask = (jnp.arange(r) > j)
            return A_in - update_A * col_mask
        
        A = jax.lax.cond(j + 1 < r, update_A_func, lambda A_in: A_in, A)
        
        w_Y = jnp.dot(v, Y_transformed)
        update_Y = tau * jnp.outer(v, w_Y)
        Y_transformed = Y_transformed - update_Y

        return A, Y_transformed

    initial_state = (A, Y_transformed)
    _, final_Y = jax.lax.fori_loop(0, r, _householder_loop_body, initial_state)
    return final_Y[r:, :]

def gls_loglik(x: jnp.ndarray, Y: jnp.ndarray,  d: jnp.ndarray, Q_C: jnp.ndarray,
               S: jnp.ndarray):
    p, s = Y.shape
    # m = B.shape[-1]
    a = 0; b = p
    # mu_m = x[a:b]; a = b; b = b + p;
    mu_p = x[a:b]; a = b; b = b + s;
    mu_s = x[a:b]
    Y = Y - jnp.outer(jnp.ones(p), (mu_s * d)) - jnp.outer(mu_p, d)
    Y = jnp.append(Q_C.T @ Y, null_space_transform_jax(Q_C, Y), axis=0)
    Y = Y.flatten('F')
    return (Y ** 2 * S).sum()
    
    

def gls_fixed_effects(data: TransformedData, 
                      error_variance: ErrorVarianceEstimates,
                      motif_variance: MotifVarianceEstimates,
                      promoter_mean: PromoterMeanEstimates, 
                      motif_mean: MotifMeanEstimates, 
                      sample_mean: SampleMeanEstimates,):
    Sigma = motif_variance.motif
    G = motif_variance.group / error_variance.variance
    d = error_variance.variance ** (-0.5)
    Y = data.Y * d
    B_decomposition = lowrank_decomposition(data.B * Sigma ** 0.5, rel_eps=None)
    n = len(B_decomposition.Q) - len(B_decomposition.S)
    S = jnp.kron(G, jnp.append(B_decomposition.S ** 2, jnp.zeros(n))) + 1
    S = 1 / S
    mu_s = sample_mean.mean.flatten() 
    mu_p = promoter_mean.mean.flatten()
    mu_m = motif_mean.mean.flatten()
    mu_p = mu_p + data.B @ mu_m
    x0 = jnp.concatenate((mu_p, mu_s))
    fun = partial(gls_loglik, Y=Y, d=d, Q_C=B_decomposition.Q,
                  S=S)
    
    fun_and_grad = jax.jit(jax.value_and_grad(fun))
    from scipy.optimize import minimize
    res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B')
    print(res)
    
    m = len(mu_m); p = len(mu_p); s = len(mu_s) 
    # a = 0; b = m; mu_m = res.x[a:b].reshape(-1, 1)
    a = 0; b = p; mu_p = res.x[a:b].reshape(-1, 1)
    a = b; b = b + s; mu_s = res.x[a:b].reshape(-1, 1)
    mu_m = (jnp.linalg.pinv(data.B) @ mu_p)
    mu_p = mu_p - data.B @ mu_m
    promoter_mean = PromoterMeanEstimates(mu_p)
    sample_mean = SampleMeanEstimates(mu_s)
    motif_mean = MotifMeanEstimates(mu_m, motif_mean.fim)

    return promoter_mean, motif_mean, sample_mean

def mle_g_loglik(G: jnp.ndarray, S: jnp.ndarray, y2: jnp.ndarray,
                 group_inds_inv: jnp.ndarray):
    G = G ** 2
    G = G[group_inds_inv]
    S = jnp.kron(G, S) + 1.0
    logdet = jnp.log(S).sum()
    S = 1 / S
    return (y2 * S).sum() + logdet


def gls_refinement(data: TransformedData, 
                   error_variance: ErrorVarianceEstimates,
                   motif_variance: MotifVarianceEstimates,
                   promoter_mean: PromoterMeanEstimates, 
                   motif_mean: MotifMeanEstimates, 
                   sample_mean: SampleMeanEstimates,):
    from scipy.optimize import minimize
    Sigma = motif_variance.motif
    d = error_variance.variance ** (-0.5)
    Y = data.Y * d
    B_decomposition = lowrank_decomposition(data.B * Sigma ** 0.5, rel_eps=None)
    B_pinv = jnp.linalg.pinv(data.B)
    n = len(B_decomposition.Q) - len(B_decomposition.S)
    S_B = jnp.append(B_decomposition.S ** 2, jnp.zeros(n))
    fun_gls = partial(gls_loglik,Y=Y, d=d, Q_C=B_decomposition.Q)
    fun_gls = jax.jit(jax.value_and_grad(fun_gls, argnums=0))
    fun_mle = partial(mle_g_loglik, S=S_B, group_inds_inv=data.group_inds_inv)
    fun_mle = jax.jit(jax.value_and_grad(fun_mle, argnums=0))
    
    
    for it in range(10):
        G = motif_variance.group / error_variance.variance
        S = jnp.kron(G, S_B) + 1
        S = 1 / S
        mu_s = sample_mean.mean.flatten() 
        mu_p = promoter_mean.mean.flatten()
        mu_m = motif_mean.mean.flatten()
        mu_p = mu_p + data.B @ mu_m
        x0 = jnp.concatenate((mu_p, mu_s))
        fun = partial(fun_gls, S=S)
        res = minimize(fun, x0, jac=True, method='L-BFGS-B')
        
        m = len(mu_m); p = len(mu_p); s = len(mu_s) 
        a = 0; b = p; mu_p = res.x[a:b].reshape(-1, 1)
        a = b; b = b + s; mu_s = res.x[a:b].reshape(-1, 1)
        
        
        Y_hat = (data.Y - mu_p - mu_s.T) * d
        Y_hat = jnp.append(B_decomposition.Q.T @ Y_hat, 
                           B_decomposition.null_space_transform(Y_hat),
                           axis=0).flatten('F') ** 2
        fun = partial(fun_mle, y2=Y_hat)
        res = minimize(fun, G ** 0.5, jac=True, method='L-BFGS-B')
        G = (res.x ** 2) * error_variance.variance
        motif_variance = MotifVarianceEstimates(motif=motif_variance.motif,
                                                group=G,
                                                fim=motif_variance.fim,
                                                fixed_group=None,
                                                loglik=motif_variance.loglik,
                                                loglik_start=motif_variance.loglik_start)
        # print(res)
        # print(G)
        
        
        mu_m = B_pinv @ mu_p
        mu_p = mu_p - data.B @ mu_m
        promoter_mean = PromoterMeanEstimates(mu_p)
        sample_mean = SampleMeanEstimates(mu_s)
        motif_mean = MotifMeanEstimates(mu_m, motif_mean.fim)

    return promoter_mean, motif_mean, sample_mean, motif_variance

def joint_refinement(data: TransformedData, 
                     error_variance: ErrorVarianceEstimates,
                     motif_variance: MotifVarianceEstimates):
    from scipy.optimize import minimize
    Sigma = motif_variance.motif
    B_decomposition = lowrank_decomposition(data.B * Sigma ** 0.5, rel_eps=None)
    Y = data.Y - data.Y.mean(axis=1, keepdims=True)
    Z = jnp.append(B_decomposition.Q.T @ Y, 
                   B_decomposition.null_space_transform(Y),
                   axis=0)
    p = len(Y)
    g = len(error_variance.variance)
    fun = partial(loglik_motifs_joint, Z=Z, g=g, 
                                      group_inds_inv=data.group_inds_inv,
                                      Sigma_hat=jnp.append(B_decomposition.S ** 2, jnp.zeros(p - len(B_decomposition.S))))
    fun = jax.jit(jax.value_and_grad(fun, argnums=0))
    x0 = jnp.append(error_variance.variance, motif_variance.group) ** 0.5
    res = minimize(fun, x0, jac=True, method='SLSQP', options={'maxiter': 1000})
    print(res)
    D = res.x[:g] ** 2
    G = res.x[g:] ** 2
    error_variance = ErrorVarianceEstimates(D, error_variance.fim, error_variance.loglik, error_variance.loglik_start)
    motif_variance = MotifVarianceEstimates(motif=motif_variance.motif,
                                            group=G,
                                            fim=motif_variance.fim,
                                            fixed_group=None,
                                            loglik=motif_variance.loglik,
                                            loglik_start=motif_variance.loglik_start)
    return error_variance, motif_variance


class ClusteringMode(str, Enum):
    none = 'none'
    KMeans = 'KMeans'
    NMF = 'NMF'

def cluster_data(B: np.ndarray, mode=ClusteringMode.none, num_clusters=200,
                 keep_motifs=False)->ProjectData:
    def trs(B, labels, n):
        mx = np.zeros((n, B.shape[1]))
        for i, v in enumerate(labels):
            mx[v, i] = 1
        return mx
    if mode == ClusteringMode.none:
        return B, None
    if mode == ClusteringMode.KMeans:
        km = KMeans(n_clusters=num_clusters, n_init=10)
        km = km.fit(B.T)
        W = km.cluster_centers_.T 
        H = trs(B, km.labels_, num_clusters); 
    else:
        model = NMF(n_components=num_clusters, max_iter=1000)
        W = model.fit_transform(B)
        H = model.components_
    if not keep_motifs:
        B = W
        clustering = H
    else:
        B = W @ H
        clustering = None
    return B, clustering

def fit(project: str, clustering: ClusteringMode,
        num_clusters: int, test_chromosomes: list, 
        gpu: bool, gpu_decomposition: bool, x64=True, true_mean=None, motif_variance: bool = True,
        promoter_variance: bool = False, test_promoters_filename: str = None,
        refinement: GLSRefinement = GLSRefinement.none, verbose=True, dump=True) -> ActivitiesPrediction:
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    group_names = data.group_names
    if clustering != clustering.none:
        logger_print('Clustering data...', verbose)
    data.B, clustering = cluster_data(data.B, mode=clustering, 
                                      num_clusters=num_clusters)
    
    if test_promoters_filename:
        with open(test_promoters_filename, 'r') as f:
            test_chromosomes = filter(lambda x: len(x), map(lambda x: x.strip(), f.readlines()))
            test_chromosomes = set(test_chromosomes)
            promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                     if p in test_chromosomes]
    elif test_chromosomes:
        import re
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')

        test_chromosomes = set(test_chromosomes)
        promoter_inds_to_drop = [i for i, p in enumerate(data.promoter_names) 
                                 if pattern.search(p).group() in test_chromosomes]
    else:
        promoter_inds_to_drop = None
    if promoter_inds_to_drop is not None:
        data.Y = np.delete(data.Y, promoter_inds_to_drop, axis=0)
        data.B = np.delete(data.B, promoter_inds_to_drop, axis=0)
    logger_print('Transforming data...', verbose)
    data_orig = transform_data(data, helmert=False)
    if gpu_decomposition:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))

    logger_print('Computing low-rank decompositions of the loading matrix...', verbose)
    with jax.default_device(device):
        B = np.append(data_orig.B, np.ones((len(data_orig.B), 1)), axis=1)
        B_decomposition_orig = lowrank_decomposition(B)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    # print(data.B.shape, data_orig.B.shape)
    with jax.default_device(device):

        logger_print('Estimating error variances...', verbose)
        error_variance = estimate_error_variance(data_orig, B_decomposition_orig, 
                                                  verbose=verbose)
        if promoter_variance:
            logger_print('Estimating FULL error variances...', verbose)
            error_variance = estimate_error_variance_full(data_orig, B_decomposition_orig, error_variance,
                                                          original_data=data_orig,
                                                          verbose=verbose)
            data_orig = transform_data(data, helmert=False, weights=error_variance.promotor ** (-0.5))
            data = transform_data(data, helmert=True, weights=error_variance.promotor ** (-0.5))
        else:
            data_orig = transform_data(data, helmert=False,)
            data = transform_data(data, helmert=True,)
        B_decomposition = lowrank_decomposition(data.B)
    
        logger_print('Estimating promoter-wise means...', verbose)
        promoter_mean = estimate_promoter_mean(data, B_decomposition,
                                               error_variance=error_variance)
        
        logger_print('Estimating variances of motif activities...', verbose)
        if motif_variance:
            motif_variance = estimate_motif_variance(data, B_decomposition,
                                                      error_variance=error_variance,
                                                      original_data=data_orig,
                                                      verbose=verbose)
        else:
            motif_variance = estimate_motif_variance_identity(data, B_decomposition,
                                                              error_variance=error_variance,
                                                              verbose=verbose)
        # logger_print('Jointly refining variances of errors and motif activities...', verbose)
        # error_variance, motif_variance = joint_refinement(data, error_variance, motif_variance)
        
        logger_print('Estimating motif means...', verbose)
        motif_mean = estimate_motif_mean(data, B_decomposition, error_variance=error_variance,
                                          motif_variance=motif_variance,
                                          promoter_mean=promoter_mean)
        logger_print('Estimating sample means...', verbose)
        sample_mean = estimate_sample_mean(data_orig, error_variance=error_variance, 
                                           motif_variance=motif_variance, motif_mean=motif_mean,
                                           promoter_mean=promoter_mean)

        # if refinement == GLSRefinement.fixed:
        #     logger_print('Refining fixed effects...', verbose)
        #     promoter_mean, motif_mean, sample_mean = gls_fixed_effects(data_orig, error_variance, motif_variance,
        #                                                                promoter_mean, motif_mean, sample_mean)
        # elif refinement == GLSRefinement.full:
        #     logger_print('Refining fixed effects and G...', verbose)
        #     promoter_mean, motif_mean, sample_mean, motif_variance = gls_refinement(data_orig, error_variance, motif_variance,
        #                                                                                promoter_mean, motif_mean, sample_mean)
    # print(promoter_mean.mean.shape, error_variance.promotor.shape)
    promoter_mean = PromoterMeanEstimates(promoter_mean.mean.flatten() * error_variance.promotor ** 0.5)
    res = FitResult(error_variance=error_variance, motif_variance=motif_variance,
                    motif_mean=motif_mean, promoter_mean=promoter_mean,
                    sample_mean=sample_mean, clustering=clustering,
                    group_names=group_names, promoter_inds_to_drop=promoter_inds_to_drop)    
    if dump:
        with openers[fmt](f'{project}.fit.{fmt}', 'wb') as f:
            dill.dump(res, f)
    return res

def split_data(data: ProjectData, inds: list) -> tuple[ProjectData, ProjectData]:
    if not inds:
        return data, None
    B_d = np.delete(data.B, inds, axis=0)
    B = data.B[inds]
    Y_d = np.delete(data.Y, inds, axis=0)
    Y = data.Y[inds]
    promoter_names_d = np.delete(data.promoter_names, inds)
    promoter_names = list(np.array(data.promoter_names)[inds])
    data_d = ProjectData(Y=Y_d, B=B_d, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names_d,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         fmt=data.fmt)
    data = ProjectData(Y=Y, B=B, K=data.K, weights=data.weights,
                         group_inds=data.group_inds, group_names=data.group_names,
                         motif_names=data.motif_names, promoter_names=promoter_names,
                         motif_postfixes=data.motif_postfixes, sample_names=data.sample_names,
                         fmt=data.fmt)
    return data_d, data

def predict(project: str, filter_motifs: bool, filter_order: int, 
            tau_search: bool, cv_repeats: int, cv_splits: int,
            tau_left: float, tau_right: float, tau_num: int, pinv: bool,
            gpu: bool, x64=True,
            dump=True):
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = read_init(project)
    fmt = data.fmt
    with openers[fmt](f'{project}.fit.{fmt}', 'rb') as f:
        fit = dill.load(f)
    data, _ = split_data(data, fit.promoter_inds_to_drop)
    data = transform_data(data, helmert=False, weights=fit.error_variance.promotor ** (-0.5))
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):
        activities = predict_activities(data, fit, tau_search=tau_search,
                                        cv_repeats=cv_repeats, cv_splits=cv_splits,
                                        tau_left=tau_left, tau_right=tau_right, tau_num=tau_num, 
                                        pinv=pinv,
                                        filter_motifs=filter_motifs, 
                                        filter_order=filter_order)
    if dump:
        with openers[fmt](f'{project}.predict.{fmt}', 'wb') as f:
            dill.dump(activities, f)
    return activities

@dataclass(frozen=True)
class FOVResult:
    total: float
    promoter: np.ndarray
    sample: np.ndarray
    
@dataclass(frozen=True)
class TestResult:
    train: tuple[FOVResult]
    test: tuple[FOVResult]
    grouped: bool

def _groupify(X: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    res = list()
    for inds in groups:
        res.append(X[:, inds].mean(axis=-1, keepdims=True))
    return np.concatenate(res, axis=-1)

def compute_mu_mle(data: TransformedData, fit: FitResult):
    U_m = fit.motif_mean.mean.reshape(-1, 1)
    mu_s = fit.sample_mean.mean.reshape(-1, 1)
    Y = data.Y - mu_s.T
    Y = Y - data.B @ U_m
    
    Sigma = fit.motif_variance.motif
    G = fit.motif_variance.group
    D = fit.error_variance.variance
    groups = data.group_inds_inv
    G = G[groups]
    D = D[groups]
    # Compute B√Σ using broadcasting
    B_tilde = data.B * jnp.sqrt(Sigma[None, :])
    
    # Economy-size SVD (p x k), k = min(p, m)
    U, s, _ = jnp.linalg.svd(B_tilde, full_matrices=False)
    s_sq = s**2
    
    # Compute residual space components first
    sum_Y_over_d = Y @ (1/D)  # \sum_i Y_i/D_ii
    sum_1_over_d = jnp.sum(1/D)  # \sum_i 1/D_ii
    
    # Projection and residual calculation
    proj = U @ (U.T @ sum_Y_over_d)
    mu_residual = (sum_Y_over_d - proj) / sum_1_over_d
    
    # Compute signal space components
    UTY = U.T @ Y  # k x s
    
    # Create inverse factor matrix (s x k)
    inv_factors = 1 / (G[:, None] * s_sq[None, :] + D[:, None])
    
    # Compute weighted sums
    sum_term1 = jnp.sum(UTY.T * inv_factors, axis=0)  # Sum over observations
    a_j = jnp.sum(inv_factors, axis=0)  # Normalization factors
    
    # Combine components
    mu_signal = U @ (sum_term1 / a_j)
    mu_hat = mu_signal + mu_residual
    
    return mu_hat

def _cor(a, b, axis=1):
    a_centered = a - a.mean(axis=axis, keepdims=True)
    b_centered = b - b.mean(axis=axis, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=axis)
    denominator = np.sqrt(np.sum(a_centered**2, axis=axis) * np.sum(b_centered**2, axis=axis))
    return numerator / denominator


def predict_mu_p_test(B: np.ndarray, Z: np.ndarray, mu_p: np.ndarray, test_inds: np.ndarray, 
                      n_B: int = 8, n_Z: int = 8, n_neighbours: int = 64) -> np.ndarray:

    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsRegressor

    if n_B > 0:
        pca_b = PCA(n_components=n_B)
        B_reduced = pca_b.fit_transform(B)
    else:
        if n_B == -1:
            B_reduced = B
        else:
            B_reduced = None
    
    if n_Z > 0:
        pca_z = PCA(n_components=n_Z)
        Z_reduced = pca_z.fit_transform(Z)
        if B_reduced is None:
            comb = [Z_reduced, ]
        else:
            comb = [B_reduced, Z_reduced]
    else:
        if n_Z == -1:
            Z_reduced = Z
            if B_reduced is None:
                comb = [Z_reduced]
            else:
                comb = [B_reduced, Z_reduced]
        else:
            Z_reduced = None
            comb = [B_reduced, ]
    
    combined_features = np.hstack(comb)
    

    p = combined_features.shape[0]
    all_indices = np.arange(p)

    train_inds = np.setdiff1d(all_indices, test_inds)
    


    X_train = combined_features[train_inds]
    y_train = mu_p
    X_test = combined_features[test_inds]
    
    

    reg = KNeighborsRegressor(n_neighbors=n_neighbours, weights='distance', )
    reg.fit(X_train, y_train)
    

    predictions = reg.predict(X_test)
    return predictions


def calculate_fov(project: str, use_groups: bool, gpu: bool, 
                  stat_type: GOFStat, stat_mode: GOFStatMode, weights: bool = True,
                  mean_mode: FOVMeanMode = FOVMeanMode.gls, knn_n=128, pca_b=64, pca_z=3,
                  x64=True, verbose=True, dump=True):
    def calc_fov(data: TransformedData, fit: FitResult,
                 activities: ActivitiesPrediction, mu_p=None, a_vec=None) -> tuple[FOVResult]:
        def sub(Y, effects) -> FOVResult:
            if stat_type == stat_type.fov:
                Y1 = Y - effects
                Y = Y - Y.mean()
                Y1 = Y1 - Y1.mean()
                Y = Y ** 2
                Y1 = Y1 ** 2
                prom = 1 - Y1.mean(axis=1) / Y.mean(axis=1)
                sample = 1 - Y1.mean(axis=0) / Y.mean(axis=0)
                total = 1 - Y1.mean() / Y.mean()
            elif stat_type == stat_type.corr:
                total = np.corrcoef(Y.flatten(), effects.flatten())[0, 1]
                prom = _cor(Y, effects, axis=1)
                sample = _cor(Y, effects, axis=0)
            return FOVResult(total, prom, sample)
        B = data.B
        drops = activities.filtered_motifs
        U_m = fit.motif_mean.mean.reshape(-1, 1)
        if mu_p is None:
            mu_p = fit.promoter_mean.mean
        mu_s = fit.sample_mean.mean.reshape(-1, 1)
        mu_p = mu_p.reshape(-1,1)
        Y = data.Y
        if weights:
            if a_vec is None:
                a_vec = fit.error_variance.promotor ** (-0.5)
                if len(a_vec) != len(Y):
                    a_vec = np.ones(len(Y))
            Y = a_vec.reshape(-1, 1) * Y
            B = a_vec.reshape(-1, 1) * B
            mu_p = a_vec.reshape(-1, 1) * mu_p
        else:
            a_vec = jnp.ones(len(Y))
        d1 = mu_p.reshape(-1, 1) + jnp.outer(a_vec, mu_s.flatten())
        d2 = B @ U_m
        d2 = d2.repeat(len(mu_s), -1)
        # Y1 = Y0 - mu_p.reshape(-1, 1) - mu_s.reshape(1, -1)
        if use_groups:
            U = activities.U
            groups = data.group_inds
            Y = _groupify(Y, groups)
            d1 = _groupify(d1, groups)
            d2 = _groupify(d2, groups)
        else:
            U = activities.U_raw
        if activities.clustering is not None:
            d3 = activities.clustering[0] @ U
        else:
            d3 = np.delete(B, drops, axis=1) @ U
        if stat_mode == stat_mode.residual:
            stat_0 = sub(Y, d1 + d2 + d3)
            stat_1 = sub(Y - d1, d2 + d3)
            stat_2 = sub(Y - d1 - d2, d3)
        elif stat_mode == stat_mode.total:
            stat_0 = sub(Y, d1)
            stat_1 = sub(Y, d1 + d2)
            stat_2 = sub(Y, d1 + d2 + d3)
        return stat_0, stat_1, stat_2
    data = read_init(project)
    fmt = data.fmt
    with openers[fmt](f'{project}.fit.{fmt}', 'rb') as f:
        fit : FitResult = dill.load(f)
    with openers[fmt](f'{project}.predict.{fmt}', 'rb') as f:
        activities : ActivitiesPrediction = dill.load(f)
    if mean_mode == mean_mode.knn:
        B0 = transform_data(data, helmert=False).B
    data, data_test = split_data(data, fit.promoter_inds_to_drop)
    if x64:
        jax.config.update("jax_enable_x64", True)
    data = transform_data(data, helmert=False, )
    if data_test is not None:
        data_test = transform_data(data_test, helmert=False)
    if gpu:
        device = jax.devices()
    else:
        device = jax.devices('cpu')
    device = next(iter(device))
    with jax.default_device(device):

        if data_test is not None:
            drops = activities.filtered_motifs
            U = activities.U_raw
            U_m = fit.motif_mean.mean.reshape(-1, 1)
            mu_s = fit.sample_mean.mean.reshape(-1, 1)
            if mean_mode == mean_mode.gls:
                Y = data_test.Y - mu_s.T
                Y = Y - data_test.B @ U_m
                Y = Y - np.delete(data_test.B, drops, axis=1) @ U
                D = (1 / fit.error_variance.variance)[data_test.group_inds_inv].reshape(-1, 1)
                mu_p = Y @ D / (D.sum()) 
            elif mean_mode == mean_mode.knn:
                mu_p = fit.promoter_mean.mean.flatten()
                B = B0
                Z = (B @ fit.motif_mean.mean.reshape(-1, 1)) + fit.sample_mean.mean.reshape(1, -1)
                Z = Z + np.delete(B, drops, axis=1) @ U
                mu_p = predict_mu_p_test(B, Z, mu_p, fit.promoter_inds_to_drop,
                                         n_neighbours=knn_n, n_Z=pca_z, n_B=pca_b)
            elif mean_mode == mean_mode.null:
                mu_p = np.zeros((len(data_test.Y), 1))

            test_FOV = calc_fov(data=data_test, fit=fit, activities=activities,
                                mu_p=mu_p)
        train_FOV = calc_fov(data=data, fit=fit, activities=activities)
    if data_test is None:
        test_FOV = None
    res = TestResult(train_FOV, test_FOV, grouped=use_groups)
    with openers[fmt](f'{project}.fov.{fmt}', 'wb') as f:
        dill.dump(res, f)
    return res
        
        
        