.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/dbopt.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/dbopt
    .. image:: https://readthedocs.org/projects/dbopt/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://dbopt.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/dbopt/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/dbopt
    .. image:: https://img.shields.io/pypi/v/dbopt.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/dbopt/
    .. image:: https://img.shields.io/conda/vn/conda-forge/dbopt.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/dbopt
    .. image:: https://pepy.tech/badge/dbopt/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/dbopt
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/dbopt

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
dbopt
=====


    Enforcing a user specified topology on the decision boundary of a neural network.


Dbopt allows the user to specify a set of Betti numbers for the decision boundary of a neural network. It then randomly samples the decision boundary to construct a filtered simplicial complex that resembles the decision boundary and allows to compute its persistent homology. The obtained Betty numbers are then compared with the desired ones, and a loss function is computed. The gradient of this loss function is then propagated back to the weights of the neural network through implicit differentiation, so the weights can be updated in order to better fit the desired homology. For more details, please refer to the [paper draft](https://github.com/Noam-Ghenassia/dbopt/blob/main/Topological_optimisation_of_the_decision_boundary.pdf)


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
