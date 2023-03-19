from gph.python import ripser_parallel
import numpy as np
from jax import numpy as jnp
from jax.lax import stop_gradient
from jax import vmap, jit

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

    def __init__(self, zeta: float = 0.5, homology_dimensions: list = [0, 1],
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

    
    
    def _computing_persistence_with_gph(self, X, N):
        """This method accepts the pointcloud and returns the
        persistence diagram in the following form
        $Pers:Filt_K \subset \mathbb R^{|K|} \to (\mathbb R^2)^p
        \times \mathbb R^q, \Phi(X) \mapsto D = \cup_i^p
        (\Phi_{\sigma_i1}(X) ,\Phi_{\sigma_i2}(X) )
        \times \cup_j^q (\Phi_{\sigma_j}(X),+\infty).$
        The persistence diagram ca be readily used for 
        gradient descent.
        Args:
            X (jnp.array):
                point cloud
        Returns:
            list of shape (n, 3):
                Persistence pairs (correspondig to the
                first 2 dimensions) where the last dimension 
                contains the homology dimension
        """

        embedding = jnp.concatenate((X, 3*N), axis=1)
        print("before ripser")
        output = ripser_parallel(np.asarray(stop_gradient(embedding)),
                                 maxdim=max(self.homology_dimensions),      # set equal to 5 for debugging
                                 thresh=self.max_edge_length,
                                 coeff=2,
                                 metric=self.metric,
                                 collapse_edges=self.collapse_edges,
                                 n_threads=-1,
                                 return_generators=True)
        print("after ripser")
        persistence_pairs = []
        for dim in self.homology_dimensions:
            if dim == 0:
                # x[1] and x[2] are the indices of the vertices at the ends of the
                # 1D-simplex that killed a connected component. Therefore, the
                # euclidean distance between them is the filtration value at which
                # this CC dies.
                
                persistence_pairs += [(0, jnp.linalg.norm(embedding[x[1]]-embedding[x[2]]),
                                      0) for x in output["gens"][dim]]
            else:
                # x[0] and x[1] are the indices of the extremities of the edge that,
                # when added, creates a new features in the homology of dimension dim.
                # Similarly, x[2] and x[3] are the indices of the edge that kills this feature.
                persistence_pairs += [(jnp.linalg.norm(embedding[x[1]]-embedding[x[0]]), 
                                      jnp.linalg.norm(embedding[x[3]]-embedding[x[2]]), 
                                      dim) for x in output["gens"][1][dim-1]]

        return persistence_pairs

@jit
def metric_with_normal_vectors(points, normal_vectors):
    
    n_points = points.shape[0]
    input_dim = points.shape[1]
    
    @jit
    def dissimilarity_scalar(x:jnp.array, y:jnp.array)-> jnp.array:
        return jnp.linalg.norm(x-y)
    dissimilarity_matrix_fn = vmap(dissimilarity_scalar, in_axes=(1, 1))
    
    a = jnp.repeat(jnp.transpose(normal_vectors)[:, :,  jnp.newaxis], n_points, axis=2)
    b = jnp.transpose(a, axes=[0, 2, 1])
    normal_vectors_distance = dissimilarity_matrix_fn(
        jnp.reshape(a, (input_dim, n_points**2)),
        jnp.reshape(b, (input_dim, n_points**2))).reshape((n_points, n_points))
    
    c = jnp.repeat(jnp.transpose(points)[:, :,  jnp.newaxis], n_points, axis=2)
    d = jnp.transpose(c, axes=[0, 2, 1])
    points_distance = dissimilarity_matrix_fn(
        jnp.reshape(c, (input_dim, n_points**2)),
        jnp.reshape(d, (input_dim, n_points**2))).reshape((n_points, n_points))
    
    #return 100*normal_vectors_distance + points_distance
    return normal_vectors_distance + points_distance

def plot_persistence_diagram(pers_diag, ax):
    
    H0 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==0])
    H1 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==1])
    diag = np.linspace(0, 11, 100)
    ax.scatter(H0[:, 0], H0[:, 1], label='H0')
    ax.scatter(H1[:, 0], H1[:, 1], label='H1')
    ax.plot(diag, diag)
    