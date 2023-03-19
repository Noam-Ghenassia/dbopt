from abc import ABC, abstractmethod
from typing import Union, Dict, Callable

from jax import numpy as jnp
from jax import nn
from jax import grad
#from jax import random
from jaxopt.implicit_diff import custom_root
import optax


from src.dbopt.DB_sampler import DecisionBoundarySampler
from src.dbopt.persistent_gradient import PersistentGradient

class DecisionBoundrayGradient():
    """This class allows to compute the gradient of the points with respect to the
    weights of the neural network that parametrize the manifold sampled by the points.
    To do this, it uses the implicit differentiation method provided in
    https://arxiv.org/abs/2105.15183. Given a sampling of the decision boundary, it
    creates a set of 1D parameters t (one parameter per point) that allow the points
    to move normally to the decision boundary. The optimality condition is then defined
    on t : since the original points already lie on the decision boundary, the optimal
    values of t are 0. This implicitely defines a function t^{\star}(theta), where theta
    are the parameters of the neural network. Jaxopt then allows to get the gradient of
    t^{\star} with respect to theta, which will then be used to compute the gradient of
    the topological loss.
    
    """
    
    def __init__(self, net: callable, sampled_points: jnp.array, with_logits=True):
        self.net: Callable[[jnp.array], jnp.array] = net
        self.sampled_points = sampled_points
        self.with_logits = with_logits
    
    def get_points(self):
        return self.sampled_points
    
    def _difference_of_the_logits(self, x, theta):
        """This function returns the difference of the logits, evaluated at the points in X.
        The gradient of the sum of its outputs gives the normal vectors of the decision
        boundary, evaluated at the points in X.

        Args:
            x (jnp.array): The points at which the function is evaluated.
            theta (jax.FrozenDict): The parameters of the network.

        Returns:
            jnp.array: The difference of the logits.
        """
        logits = self.net.apply(theta, x)
        return logits[:, 1] - logits[:, 0]
    
    def _normal_unit_vectors(self, theta: Union[jnp.array, Dict[str, jnp.array]]):
        """This function returns a set of vectors that are normal to the decision boundary
        at the points that were sampled from it by the sampler.

        Args:
            theta (jnp.array): the weights of the neural network

        Returns:
            jnp.array: an n*d matrix with rows the normal vectors of the decision boundary
            evaluated at the points sampled by the sampler.
        """
        if self.with_logits:
            normal_vectors = grad(lambda x : self._difference_of_the_logits(x, theta).sum())(self.sampled_points)
        else:
            normal_vectors = grad(lambda x : self.net.apply(theta, x).sum())(self.sampled_points)
        norms = jnp.linalg.norm(normal_vectors, axis=1).reshape(-1, 1)
        return normal_vectors / norms
    
    def _parametrization_normal_lines(self, t: jnp.array, theta: jnp.array):
        """This function gives a new set of points that depend on the points initially sampled
        by the sampler, but that only depend on a single real coordinate each. These new
        points are bound to move along a line that is normal to the decision boundary
        and passes through one of the original points (each). When t=0, the returned points
        equal the old ones.

        Args:
            t (jnp.array): the parameters that define the position of the new points.
            theta (jnp.array): the weights of the neural network

        Returns:
            jnp.array: the new points.
        """

        return self.sampled_points + jnp.expand_dims(t, 1) * self._normal_unit_vectors(theta)


    def _optimality_condition(self, t, theta):
        """This function is the optimality condition that implicitly defines
        t^{\star}(theta) : for a given theta, the optimality condition is zero
        when the points lie on the decision boundary.

        Args:
            t (jnp.array): the coordinates of the new points along the normal vectors. In
            the setting of jaxopt, these are the variables optimized in the inner
            optimization problem.
            theta (jnp.array): the parameters of the network.

        Returns:
            float: the points loss, i.e., the sum of the squared distances to the DB
        """
        points_along_normal_lines = self._parametrization_normal_lines(t, theta)
        logits = self.net.apply(theta, points_along_normal_lines)
        
        if logits.ndim == 2:
            deviation_from_decision_boundary = logits[:, 0]-logits[:, 1]
        else :
            deviation_from_decision_boundary = logits
        return deviation_from_decision_boundary
    
    
    def _inner_problem(self, t_init, theta):
        """This function is the inner optimization problem. It returns the
        optimal t vector that satisfies the optimality condition (namely,
        the 0 vector).

        Args:
            t_init (jnp.array): this parameter is not used, but is necessary for jaxopt.
            theta (jnp.array): the parameters of the network.
        """

        del t_init
        return jnp.zeros(self.sampled_points.shape[0])
    
        
    def t_star(self, theta):
        """ This function wraps the inner problem with the jaxopt function that allows
        to compute the gradient with respect to the parameters of the network.
        
        Args:
            theta (jnp.array): the parameters of the network.
        """
        t_init = None
        return custom_root(self._optimality_condition)\
            (self._inner_problem)(t_init, theta)

class TopologicalLoss():
    """This is the abstract class from which the different topological losses are inherited.
    """

    def __init__(self, net, sampled_points, desired_homology: dict[int:jnp.array],
                 with_logits=True, inflate_desired_features=True):
        """
        Args:
            net (Callable): the network that is being optimized.
            sampled_points (jnp.array): A sampling of the network's decision boundary.
            desired_homology (dict[int:jnp.array]): a dictionaries with keys the homology
                dimensions of interest and values the corresponding desired Betti number.
        """
        self.sampled_points = sampled_points
        self.desired_homology = desired_homology
        self.inflate_desired_features = inflate_desired_features
        self.persistent_gradient = PersistentGradient(homology_dimensions=list(desired_homology.keys()))
        self.db_grad = DecisionBoundrayGradient(net, sampled_points, with_logits=with_logits)
    
    def _toploss(self, parametrized_sampling, normal_vectors):

        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling, normal_vectors)
        H_i = {dim:jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==dim])
               for dim in list(self.desired_homology.keys())}
        lifetimes = {dim:hom[:, 1]-hom[:, 0] for (dim, hom) in H_i}
        desired_indices = {dim:jnp.argpartition(l, self.desired_homology[dim])[-self.desired_homology[dim]] for
                   (dim, l) in lifetimes}
        
        if self.inflate_desired_features:
            desired = {dim:lifetimes[dim][desired_indices[dim]] for dim in list(lifetimes.keys())}
            others = {dim:jnp.delete(lifetimes, desired_indices[dim]) for dim in list(lifetimes.keys())}
            losses = jnp.array([jnp.sum(others[dim]**2) - jnp.sum(desired[dim]**2) for dim in list(lifetimes.keys())])
            return jnp.sum(losses)
        
        others = {dim:jnp.delete(lifetimes, desired_indices[dim]) for dim in list(lifetimes.keys())}
        losses = jnp.array([jnp.sum(others[dim]**2) for dim in list(lifetimes.keys())])
        

    
    def update_sampled_points(self, new_points):
        """This function allows to provide a new sampling of the decision boundary.
        It should be called each time the network's decision boundary changes, so
        the class can reliably keep track of its topology.

        Args:
            new_points (jnp.array): An updated sampling of the decision boundary.
        """
        self.sampled_points = new_points
        
    
    def differentiable_topological_loss(self, theta):
        """This function calls the specific topological loss indicated by the user,
        and computes its value at the sampled points, while allowing to backpropagate
        back to the weights of the network.

        Args:
            theta (jnp.array): The parameters of the network.

        Returns:
            jnp.array: The topological loss value.
        """

        t = self.db_grad.t_star(theta)
        parametrized_sampling = self.db_grad._parametrization_normal_lines(t, theta)
        normal_vectors = self.db_grad._normal_unit_vectors(theta)
        return self._toploss(parametrized_sampling, normal_vectors)

######################################################################
######################################################################
    

class DecisionBoundrayOptimizer():
    """This class allows to optimize the network's decision boundary, by
    performing gradient descent with the gradients of the topological loss.
    """
    
    
    def __init__(self, net, theta, n_sampling, desired_homology:dict[int, jnp.array],
                 input_dimension, inflate_desired_features=True, sampling_epochs=1000,
                 update_epochs=3, sampling_lr=0.01, optimization_lr=0.01, min=-1, max=1.,
                 with_logits=True, with_dataset=True):
        """Args:
            net (Callable): The network that is being optimized.
            theta (jnp.array): The parameters of the network.
            n_sampling (int): The number of points to sample the decision boundary.
            sampling_epochs (int, optional): The number of sampling epochs. Defaults to 1000.
            update_epochs (int, optional): The number of sampling epochs between optimization epochs. Defaults to 3.
            sampling_lr (float, optional): The learning rate of the sampler. Defaults to 0.01.
            optimization_lr (float, optional): The learning rate applied for the optimization. Defaults to 0.01.
        """
        self.net = net
        self.theta = theta
        self.use_cross_entropy_loss = with_dataset
        self.update_epochs = update_epochs
        self.optimization_lr = optimization_lr
        self.sampler = DecisionBoundarySampler(n_points=n_sampling, input_dim=input_dimension, min=min, max=max)
        self.sampled_points = self.sampler.sample(theta, net, points=None, lr=sampling_lr, epochs=sampling_epochs)
        self.toploss = TopologicalLoss(self.net, self.sampled_points, desired_homology=desired_homology,
                                       with_logits=with_logits, inflate_desired_features=inflate_desired_features)
    
    def _update_sampled_points(self):
        """This function updates the sampled points so they remain on the decsion boundary
        after it was updated. It is called after each optimization step.
        """
        new_points = self.sampler.sample(self.theta, self.net, self.sampled_points,
                                         epochs=self.update_epochs, delete_outliers=False)
        self.sampled_points = new_points
        self.toploss.update_sampled_points(new_points)
    
    # def make_batches(self, key, dataset, batch_size=64):                    #TODO: make the training compatible with batches
    #     """This function allows to partition the dataset into batches with
    #     specified size. 

    #     Args:
    #         dataset (jnp.array): The dataset that is partitionned.
    #         batch_size (int, optional): The size of the returned batches. Defaults to 64.

    #     Returns:
    #         tuple: The batches.
    #     """
    #     n_points = dataset.shape[0]
    #     remainder = n_points % batch_size
    #     num_full_batches = (n_points - remainder)/batch_size
    #     permuted_dataset = random.permutation(key, dataset)
    #     batches_list = []
    #     for batch in range(int(num_full_batches)):
    #         batches_list.append(permuted_dataset[batch:batch+batch_size, :])
    #     batches_list.append(permuted_dataset[-remainder-1:-1, :])
    #     return batches_list
    
    def make_loss_fn(self, data, labels):
        """This function allows to create a categorical cross-entropy
        loss function that is evaluated over a given set of data points
        and corresponding labels.

        Args:
            data (jnp.array): The datapoints.
            labels (jnp.array): The corresponding labels.
            
        Returns:
            Callable: the cross-entropy loss function.
        """
        
        def loss_fn(params):
            preds = self.net.apply(params, data)
            one_hot_gt_labels = nn.one_hot(labels, num_classes=2)
            loss = -jnp.mean(jnp.sum(one_hot_gt_labels * jnp.log(preds), axis=-1))
            return loss
        
        return loss_fn


    def optimize(self, n_epochs, dataset=None):
        """This function allows to optimize the decision boundary. It uses the Adam
        optimizer to minimize the value of the topological loss. After each epoch,
        it updates the sampling of the decision boundary by calling the sample method
        of the DecisionBoundarySampler with the current points as initial points.

        Args:
            n_epochs (int): The number of optimization epochs.

        Returns:
            jnp.array: the parameters of the network after optimization.
        """
        
        params = self.theta
        optimizer = optax.adam(self.optimization_lr)
        opt_state = optimizer.init(params)
        
        if self.use_cross_entropy_loss:
            data = dataset[:, 1:]
            labels = jnp.squeeze(dataset[:, :1], axis=1)
            CE_loss = self.make_loss_fn(data, labels)
            loss = lambda x: self.toploss.differentiable_topological_loss(x)+2000*CE_loss(x)
        else:
            loss = lambda x: self.toploss.differentiable_topological_loss(x)

        for epoch in range(n_epochs):
            print('epoch : ', epoch)
            grads = grad(loss)(self.theta)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.theta = params
            self._update_sampled_points()
        
        return self.theta
    
    def get_points(self):
        return self.sampled_points