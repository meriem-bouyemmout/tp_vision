from __future__ import print_function
import numpy as np
import random 
import matplotlib.pyplot as plt
from imageManip import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # fixer les dimensions par défaut des figures
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    ### VOTRE CODE ICI - DEBUT (remplacez l'instruction 'pass' par votre code)
    transpose_b = np.transpose(b)
    transpose_a = np.transpose(a)
    out = np.dot(transpose_b,transpose_a)
    ### VOTRE CODE ICI - FIN


    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    ### VOTRE CODE ICI - DEBUT
    pass
    ### VOTRE CODE ICI - FIN

    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    ### VOTRE CODE ICI - DEBUT
    pass
    ### VOTRE CODE ICI - FIN

    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
    ### VOTRE CODE ICI - DEBUT
    pass
    ### VOTRE CODE ICI - FIN
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (1, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    ### VOTRE CODE ICI - DEBUT
    pass
    ### VOTRE CODE ICI - FIN
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    ### VOTRE CODE ICI - DEBUT
    pass
    ### VOTRE CODE ICI - FIN
    return eigenvalues, eigenvectors


M = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]]) 
a = np.array([[1, 1, 0]])
b = np.array([[-1],
            [2],
            [5]])
### VOTRE CODE ICI - FIN



print("M = \n", M)
print("Dimension de M : ", M.shape)
print()
print("a = ", a)
print("Dimension de a : ", a.shape)
print()
print("b = ", b)
print("Dimension de b : ", b.shape)


aDotB = dot_product(a, b)
print(aDotB)

print("dimension : ", aDotB.shape)