import numpy as np
from scipy.sparse import csr_array
from pycvxcluster.algos.helpers import prox_l2
from pycvxcluster.algos.helpers import proj_l2
from pycvxcluster.algos.helpers import fnorm
from sksparse.cholmod import cholesky
from scipy.sparse import eye
import scipy.sparse.linalg as sla
import time


def admm_l2(
    X,
    A,
    weight_vec,
    max_iter=20000,
    sigma=1,
    rho=1.618,
    stop_tol=1e-6,
    verbose=1,
    xi0=None,
    x0=None,
    y0=None,
):
    if verbose > 0:
        print("Starting ADMM...")
    start_time = time.perf_counter()

    d, n = X.shape
    E = len(weight_vec)

    factor = cholesky(eye(n) + sigma * A @ A.T)

    msg = ""
    if xi0 is None:
        xi = X.copy()
    else:
        xi = xi0
    if y0 is None or x0 is None:
        y = csr_array((d, E))
        x = y
    else:
        y = y0
        x = x0
    Axi = xi @ A
    AtAxi = Axi @ A.T
    Atx = x @ A.T
    Rp = Axi - y
    proj_x = proj_l2(x, weight_vec)
    Rd = x - proj_x
    primfeas = fnorm(Rp) / (1 + fnorm(y))
    dualfeas = fnorm(Rd) / (1 + fnorm(x))
    maxfeas = max(primfeas, dualfeas)

    breakyes = 0
    prim_win = 0
    dual_win = 0

    for iter in range(max_iter):
        rhsxi = X - Atx + sigma * y @ A.T
        rhsxi = rhsxi.T
        xit = np.zeros_like(xi).T
        for i in range(xit.shape[1]):
            xit[:, i] = factor(rhsxi[:, i])
        xi = xit.T
        Axi = xi @ A
        yinput = Axi + (1 / sigma) * x
        y, rr, _ = prox_l2(yinput, weight_vec / sigma)
        Rp = Axi - y
        x = x + rho * sigma * Rp
        Atx = x @ A.T
        normRp = fnorm(Rp)
        normy = fnorm(y)
        primfeas = normRp / (1 + normy)
        proj_x = proj_l2(x, weight_vec)
        Rd = x - proj_x
        dualfeas = fnorm(Rd) / (1 + fnorm(x))
        maxfeas = max(primfeas, dualfeas)
        if maxfeas < stop_tol:
            primobj = 0.5 * fnorm(xi - X) ** 2 + np.sum(
                weight_vec * np.sqrt(np.sum(Axi * Axi, axis=0))
            )
            dualobj = -0.5 * fnorm(Atx) ** 2 + np.sum(X * Atx)
            relgap = np.abs(primobj - dualobj) / (1 + np.abs(primobj) + np.abs(dualobj))
            eta = relgap

            if eta < stop_tol:
                breakyes = 1
                msg = "Successful convergence"
                break
    if breakyes == 0:
        msg = "Maximum number of iterations reached"
    primobj = 0.5 * fnorm(xi - X) ** 2 + np.sum(
        weight_vec * np.sqrt(np.sum(Axi * Axi, axis=0))
    )
    dualobj = -0.5 * fnorm(Atx) ** 2 + np.sum(X * Atx)
    relgap = np.abs(primobj - dualobj) / (1 + np.abs(primobj) + np.abs(dualobj))
    eta = relgap
    end_time = time.perf_counter()
    if verbose > 0:
        print("ADMM finished in {} seconds.".format(end_time - start_time))
        print(f"Termination status: {msg}, iterations: {iter}")
    return (
        xi,
        y,
        x,
        breakyes,
        iter,
        eta,
        primfeas,
        dualfeas,
        primobj,
        dualobj,
        end_time - start_time,
    )
