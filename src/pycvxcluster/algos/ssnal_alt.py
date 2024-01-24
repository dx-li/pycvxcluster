import numpy as np

# import pyproximal
import pylops

# import logging
import scipy
import scipy.sparse
import scipy.sparse.linalg as sla

# from numba import jit, njit, prange

# Preliminaries


# @jit(nopython=True, parallel=True, fastmath=True)
# def sum_with_numba(X, hat_r, hat_c, D, alpha_less_1):
#     return np.sum((X[:, hat_r] - X[:, hat_c]) * D[:, alpha_less_1], axis=0)


class B(pylops.LinearOperator):
    r"""
    B operator using node-arc incidence matrices as defined in the paper.
    """

    def __init__(self, map, dtype=None):
        self.map = map
        # self.mapT = self.map.transpose()
        super().__init__(dtype=np.dtype(dtype), shape=(map.shape[0], map.shape[1]))

    def _matmat(self, X):
        # return right_multiply(X, self.map.data, self.map.indices, self.map.indptr, self.map.shape)
        return X @ self.map

    def _adjoint(self, Z):
        # return right_multiply_transpose(Z, self.map.data, self.map.indices, self.map.indptr, self.map.shape)
        return Z @ self.map.T


def fnorm(X):
    if scipy.sparse.issparse(X):
        return sla.norm(X, "fro")
    return np.linalg.norm(X, "fro")


def fnormsq(X):
    return fnorm(X) ** 2


def matinner(X, Y):
    return np.trace(np.inner(X, Y))


def column_norms(X):
    return np.sqrt(np.einsum("ij,ij->j", X, X))


class SSNAL:
    r""" """

    def __init__(self, A, weights_matrix, gamma):
        # Initialization
        self.A = A
        self.d, N = A.shape
        self.n = weights_matrix.shape[0]
        self.nz_r, self.nz_c = np.nonzero(np.triu(weights_matrix))
        self.weights = gamma * weights_matrix[self.nz_r, self.nz_c]
        self.E = len(self.nz_r)
        # self.J = np.zeros((self.n,self.n))
        # self.J[self.nz_r, self.nz_c] = 1
        # self.J[self.nz_c, self.nz_r] = 1
        cJ = scipy.sparse.lil_array((self.n, self.E))
        cJ[self.nz_r, np.arange(self.E)] = 1
        # cJ = cJ.tocsr()
        # cJ_bar = scipy.sparse.lil_array((self.n, self.E))
        cJ[self.nz_c, np.arange(self.E)] = -1
        cJ = cJ.tocsr()
        self.Bop = B(cJ)

    def pU(self, U):
        return np.dot(self.weights, column_norms(U))

    def prox_pU(self, U, tau=1):
        weights = self.weights
        upper = tau * weights
        norms = column_norms(U)
        norms = np.maximum(norms, upper)
        return U * (1 - upper / norms)

    def proxdual_pU(self, U, tau=None):
        weights = self.weights
        upper = weights
        norms = column_norms(U)
        norms = np.maximum(norms, upper)
        return U * (upper / norms)

    def primal(self, X, U):
        return 0.5 * fnormsq(X - self.A) + self.pU(U)

    def dual(self, X, U, Z):
        adZ = self.Bop._adjoint(Z)
        return -0.5 * fnormsq(adZ) + matinner(self.A, adZ)

    def kkt_1(self, V, X):
        return V + X - self.A

    def kkt_2(self, U, Z):
        return U - self.prox_pU(U + Z)

    def kkt_3(self, X, U):
        return self.Bop._matmat(X) - U

    def kkt_4(self, Z, V):
        return self.Bop._adjoint(Z) - V

    def kkt_residual(self, X, U, Z):
        etp = (fnorm(self.kkt_3(X, U))) / (1 + fnorm(U))

        etd = fnorm(Z - self.proxdual_pU(Z)) / (1 + fnorm(Z))

        et = fnorm(self.kkt_4(Z, -X + self.A)) / (1 + fnorm(X)) + fnorm(
            self.kkt_2(U, Z)
        ) / (1 + fnorm(U))

        return etp, etd, et

    def lagrangian(self, X, U, Z):
        return 0.5 * fnormsq(X - self.A) + self.pU(U) + matinner(Z, self.kkt_3(X, U))

    def augmented_lagr(self, X, U, Z, sigma):
        return self.lagrangian(X, U, Z) + sigma / 2 * (fnormsq(self.kkt_3(X, U)))

    def phi(self, X, Z, sigma):
        return (
            0.5 * fnormsq(X - self.A)
            + self.pU(self.prox_pU(self.Bop._matmat(X) + Z / sigma, 1 / sigma))
            + 0.5
            * fnormsq(self.proxdual_pU(sigma * self.Bop._matmat(X) + Z, sigma))
            / sigma
            - 0.5 * fnormsq(Z) / sigma
        )

    def grad_phi(self, X, Z, sigma):
        return (
            X
            - self.A
            + self.Bop._adjoint(self.proxdual_pU(sigma * self.Bop._matmat(X) + Z))
        )

    @staticmethod
    def sigma_update(iter):
        sigma_update_iter = 2
        if iter < 20:
            sigma_update_iter = 3
        elif iter < 200:
            sigma_update_iter = 3
        elif iter < 500:
            sigma_update_iter = 10
        return sigma_update_iter

    # SSNAL
    def fit(self, max_iter, sigma0=1, eps=1e-6, X0=None, U0=None, Z0=None):
        if X0 is None:
            Xi = self.A.copy()
        else:
            Xi = X0
        if U0 is None:
            Ui = self.Bop._matmat(Xi)
        else:
            Ui = U0
        if Z0 is None:
            Zi = np.zeros_like(Ui)
        else:
            Zi = Z0
        sigmai = sigma0

        termination = 1

        for iter in range(max_iter):
            # Step 1
            Xi = self.ssncg(Xi, Ui, Zi, sigmai)
            Ui = self.prox_pU(self.Bop._matmat(Xi) + Zi / sigmai, 1 / sigmai)
            # Step 2

            Zi = Zi + sigmai * self.kkt_3(Xi, Ui)
            # Check stopping criterion
            kkt_residual = self.kkt_residual(Xi, Ui, Zi)
            print("kkt_residual: ", kkt_residual)
            prim = self.primal(Xi, Ui)
            dual = self.dual(Xi, Ui, Zi)
            prim_dual_gap = np.abs(prim - dual) / (1 + np.abs(prim) + np.abs(dual))
            print("prim_dual_gap: ", prim_dual_gap)
            if np.max(kkt_residual) < eps:
                termination = 0
                break
            # Step 3
            sigmai = self.sigma_update(iter)

        return Xi, Ui, Zi, termination

    # SSNCG
    def ssncg(
        self,
        X0,
        U,
        Z,
        sigma,
        mu=0.5,
        tau=0.5,
        eta=1,
        delta=0.9,
        max_iter=40,
        tolerance=1e-6,
        cg_max_iter=300,
    ):
        # Initialization
        Xj = X0
        sigmainv = 1 / sigma
        mj = 0
        mjs = np.zeros(max_iter)
        score = 0
        for iter in range(max_iter):
            # print('ssncg iter: ', iter)
            # Step 1
            D = self.Bop._matmat(Xj) + Z / sigma
            upper = self.weights
            norms_D = column_norms(D)
            zet = upper / norms_D
            zet_less_one = zet < 1
            # zet_less_one_explicit = np.arange(self.E)[zet_less_one]
            norms_D_hat = norms_D[zet_less_one]
            D_sel = D[:, zet_less_one] / norms_D_hat

            alpha = self.weights[zet_less_one] / (sigma * norms_D_hat)

            # nz_norms_D = norms_D.nonzero()
            # alpha = np.empty(self.E)
            # alpha.fill(np.inf)
            # alpha[nz_norms_D] = (
            #     sigmainv * self.weights[nz_norms_D] / norms_D[nz_norms_D]
            # )
            # alpha_less_1 = alpha < 1
            # alpha_less_1_explicit = np.arange(self.E)[alpha_less_1]
            # hat_r = self.nz_r[alpha_less_1]
            # hat_c = self.nz_c[alpha_less_1]
            # hat_alpha = alpha[alpha_less_1]

            # # M = np.zeros((self.n, self.n))
            # # M[hat_r, hat_c] = 1 - hat_alpha
            # # M[hat_c, hat_r] = 1 - hat_alpha

            # Mij = np.zeros(self.E)
            # Mij[alpha_less_1] = 1 - hat_alpha

            # indices = np.arange(self.E)[alpha_less_1]
            # cM = scipy.sparse.lil_array((self.n, self.E))
            # cM[hat_r, indices] = 1 - hat_alpha
            # # cM = cM.tocsr()
            # # cM_bar = scipy.sparse.lil_array((self.n, self.E))
            # cM[hat_c, indices] = -(1 - hat_alpha)
            # cM = cM.tocsr()
            # cM_map = cM - cM_bar

            def V(X):
                # Y = X @ (cM)
                # Y = Mij * self.Bop._matmat(X)
                # BadjY = self.Bop._adjoint(Y)
                # aux_X = X[:, hat_r] - X[:, hat_c]
                # aux_D = D[:, alpha_less_1]
                # products = np.sum((X[:, hat_r] - X[:, hat_c]) * D[:, alpha_less_1], axis=0)
                # products = sum_with_numba(X, hat_r, hat_c, D, alpha_less_1_explicit)
                # rho = np.zeros(self.E)
                # rho[alpha_less_1] = hat_alpha / (norms_D[alpha_less_1] ** 2) * products
                # W = rho * D
                # BadjW = self.Bop._adjoint(W)

                map_X = self.Bop._matmat(X)
                map_X_hat = map_X[:, zet_less_one]
                rho = np.einsum("j, ij,ij->j", alpha, map_X_hat, D_sel)
                map_X[:, zet_less_one] = map_X_hat * alpha - D_sel * rho
                return X + sigma * self.Bop._adjoint(map_X)

            grad_phi_Xj = self.grad_phi(Xj, Z, sigma)
            cg_tolerance = fnorm(grad_phi_Xj) ** (1 + tau)
            dj = np.zeros_like(Xj)
            residual = -grad_phi_Xj
            sd = residual
            beta = 0
            for j in range(cg_max_iter):
                # print('cg iter: ', j)
                if j != 0:
                    beta = np.sum(np.square(residual)) / ssres
                    sd = residual + beta * sd
                Vsd = V(sd)
                sdVsd = np.sum(sd * Vsd)
                ssres = np.sum(np.square(residual))
                dj = dj + ssres / sdVsd * sd
                residual = residual - ssres / sdVsd * Vsd

                stopping = fnorm(grad_phi_Xj + V(dj))
                # print('stopping: ', stopping)
                if stopping <= min(eta, cg_tolerance):
                    break
            # Step 2
            # print('step 2')
            alpj = 1

            phiXj = self.phi(Xj, Z, sigma)
            innergpXjdj = matinner(grad_phi_Xj, dj)
            # mj = 0
            # while (
            #         self.phi(Xj + delta**mj * dj, Z, sigma)
            #         > phiXj + mu * delta**mj * innergpXjdj
            #     ):
            #     mj += 1
            mj = 0
            alpj = 1
            while self.phi(Xj + alpj * dj, Z, sigma) > phiXj + mu * alpj * innergpXjdj:
                alpj *= delta
            # if (self.phi(Xj + delta**mj * dj, Z, sigma)
            #     > phiXj + mu * delta**mj * innergpXjdj):
            #     while (
            #         self.phi(Xj + delta**mj * dj, Z, sigma)
            #         > phiXj + mu * delta**mj * innergpXjdj
            #     ):
            #         mj += 1
            # else:
            #     while (
            #         self.phi(Xj + delta**mj * dj, Z, sigma)
            #         <= phiXj + mu * delta**mj * innergpXjdj
            #     ):
            #         mj -= 1
            #         if mj == -1:
            #             break
            #     mj += 1
            # print('mj: ', mj)
            # mjs[iter] = mj
            # print(score)
            # if iter > 0:
            #     if mjs[iter] == mjs[iter - 1]:
            #         score += 1
            #         if score > 9:
            #             break
            #     else:
            #         score = 0
            # alpj = delta**mj

            # Step 3
            Xj_prev = Xj
            Xj = Xj + alpj * dj
            if alpj < 1e-10:
                break
            if fnorm(self.grad_phi(Xj, Z, sigma)) <= tolerance:
                break

        return Xj
