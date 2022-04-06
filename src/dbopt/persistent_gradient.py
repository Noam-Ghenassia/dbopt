from gph.python import ripser_parallel
import numpy as np

class PersistentGradient():
    '''This class computes the gradient of the persistence
    diagram with respect to a point cloud. The algorithms has
    first been developed in https://arxiv.org/abs/2010.08356 .
    Discalimer: this algorithm works well for generic point clouds.
    In case your point cloud has many simplices with same
    filtration values, the matching of the points to the persistent
    features may fail to disambiguate.
    Args:
        zeta (float): 
            the relative weight of the regularisation part
            of the `persistence_function`
        homology_dimensions (tuple): 
            tuple of homology dimensions
        collapse_edges (bool, default: False): 
            whether to use Collapse or not. Not implemented yet.
        max_edge_length (float or np.inf): 
            the maximum edge length
            to be considered not infinity
        approx_digits (int): 
            digits after which to trunc floats for
            list comparison
        metric (string): either `"euclidean"` or `"precomputed"`. 
            The second one is in case of X being 
            the pairwise-distance matrix or
            the adjaceny matrix of a graph.
        directed (bool): whether the input graph is a directed graph
            or not. Relevant only if `metric = "precomputed"`
        
    '''

    def __init__(self, zeta: float = 0.5, homology_dimensions: tuple = (0, 1),
                 collapse_edges: bool = False, max_edge_length: float = np.inf,
                 approx_digits: int = 6, metric: str = "euclidean",
                 directed: bool = False):

        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        self.metric = metric
        self.directed = directed
        self.approx_digits = approx_digits
        self.zeta = zeta
        self.homology_dimensions = homology_dimensions

    
    
    def _computing_persistence_with_gph(self, X):
        """This method accepts the pointcloud and returns the
        persistence diagram in the following form
        $Pers:Filt_K \subset \mathbb R^{|K|} \to (\mathbb R^2)^p
        \times \mathbb R^q, \Phi(X) \mapsto D = \cup_i^p
        (\Phi_{\sigma_i1}(X) ,\Phi_{\sigma_i2}(X) )
        \times \cup_j^q (\Phi_{\sigma_j}(X),+\infty).$
        The persistence diagram ca be readily used for 
        gradient descent.
        Args:
            X (np.array): 
                point cloud
        Returns:
            list of shape (n, 3):
                Persistence pairs (correspondig to the
                first 2 dimensions) where the last dimension 
                contains the homology dimension
        """
        output = ripser_parallel(X.detach().numpy(),
                                 maxdim=max(self.homology_dimensions),
                                 thresh=self.max_edge_length,
                                 coeff=2,
                                 metric=self.metric,
                                 collapse_edges=self.collapse_edges,
                                 n_threads=-1,
                                 return_generators=True)

        persistence_pairs = []
        #print(output["gens"])
        for dim in self.homology_dimensions:
            if dim == 0:
                # x[1] and x[2] are the indices of the vertices at the ends of the
                # 1D-simplex that killed a connected component. Therefore, the
                # euclidean distance between them is the filtratio value at which
                # this CC dies.
                persistence_pairs += [(0, torch.norm(X[x[1]]-X[x[2]]),
                                      0) for x in output["gens"][dim]]
            else:
                # x[0] and x[1] are the indices of the extremities of the edge that,
                # when added, creates a new features in the homology of dimension dim.
                # Similarly, x[2] and [3] are the indices of the edge that kills this feature.
                persistence_pairs += [(torch.norm(X[x[1]]-X[x[0]]), 
                                      torch.norm(X[x[3]]-X[x[2]]), 
                                      dim) for x in output["gens"][1][dim-1]]
        return persistence_pairs
    
    """def persistence_function(self, X: torch.Tensor) -> torch.Tensor:
        This is the Loss function to optimise.
        $L=-\sum_i^p |\epsilon_{i2}-\epsilon_{i1}|+
        \lambda \sum_{x in X} ||x||_2^2$
        It is composed of a regularisation term and a
        function on the filtration values that is (p,q)-permutation
        invariant.
        out = 0

        persistence_array = self._computing_persistence_with_gph(X)
        for item in persistence_array:
            out += item[1]-item[0]
        reg = (X**2).sum()  # regularisation term
        return -out + self.zeta*reg  # maximise persistence"""
