from sklearn.base import BaseEstimator, ClusterMixin
from pycvxcluster.helpers.compute_weights import compute_weights
from pycvxcluster.helpers.find_clusters import find_clusters
from pycvxcluster.helpers.ssnal import ssnal
from pycvxcluster.helpers.ssnal import AInput
from pycvxcluster.helpers.ssnal import Dim


class SSNAL(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        k,
        phi=0.5,
        gamma=1,
        clustertol=1e-5,
        sigma=1,
        maxiter=1000,
        stoptol=1e-6,
        ncgtolconst=0.5,
        verbose=1,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors to use in the graph construction.
        phi : float, optional
            Parameter for the weight matrix. The default is .5.
        gamma : float, optional
            Parameter for the weight matrix. The default is 1.
        clustertol : float, optional
            Tolerance for deciding if data points are in the same cluster. The default is 1e-5.
        sigma : float, optional
            Parameter for the objective function. The default is 1.
        maxiter : int, optional
            Maximum number of iterations. The default is 1000.
        stoptol : float, optional
            Tolerance for the stopping criterion. The default is 1e-6.
        ncgtolconst : float, optional
            Constant for the stopping criterion in ssncg. The default is 0.5.
        verbose : int, optional
            Verbosity level (0, 1, or 2). The default is 1.
        **kwargs : dict
            Keyword arguments for the SSNAL algorithm.
        """
        self.k = k
        self.phi = phi
        self.gamma = gamma
        self.clustertol = clustertol
        self.sigma = sigma
        self.maxiter = maxiter
        self.stoptol = stoptol
        self.ncgtolconst = ncgtolconst
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, y=None, return_centers=False):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self
        """
        (
            self.weight_vec_,
            self.node_arc_matrix_,
            self.weight_matrix_,
            t1,
        ) = compute_weights(X.T, self.k, self.phi, self.gamma, self.verbose)
        AI = AInput(self.node_arc_matrix_)
        dim = Dim(X.T, self.weight_vec_)
        (
            self.primobj_,
            self.dualobj_,
            _,
            xi,
            _,
            self.eta_,
            _,
            self.iter_,
            self.termination_,
            t2,
        ) = ssnal(
            AI,
            X.T,
            dim,
            self.weight_vec_,
            self.sigma,
            self.maxiter,
            self.stoptol,
            self.ncgtolconst,
            self.verbose,
            **self.kwargs,
        )
        if return_centers:
            self.centers_ = xi
        self.labels_, self.n_clusters_ = find_clusters(xi, self.clustertol)
        self.tot_time_ = t1 + t2
        if self.verbose > 0:
            print(f"Clustering completed in {self.tot_time_} seconds.")
        return self

    def fit_predict(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
