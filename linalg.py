from __future__ import print_function
import numpy as np
import random 
import matplotlib.pyplot as plt
from imageManip import *
from linalg import *



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
    
    ### VOTRE CODE ICI - DEBUT
    out1 = np.dot(b.T, a.T)
    out2 = np.dot(M, a.T)
    out = out1* out2
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
    
    ### VOTRE CODE ICI - DEBUT
    u, s, v = np.linalg.svd(M, full_matrices=False)
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
    ### VOTRE CODE ICI - DEBUT
    u, s, v = svd(M)
    singular_values = s[:k]
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
    
    ### VOTRE CODE ICI - DEBUT
    w, v = np.linalg.eig(M)
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
    ### VOTRE CODE ICI - DEBUT
    w, v = eigen_decomp(M)
   
    indices_triés = np.argsort(w)[::-1]
    top_indices = indices_triés[:k]

    eigenvalues = w[top_indices]
    eigenvectors = v[:, top_indices]


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

N = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]) 


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


ans = complicated_matrix_function(M, a, b)
print(ans)
print()
print("dimension : ", ans.shape)


M_2 = np.array(range(4)).reshape((2,2))
a_2 = np.array([[1,1]])
b_2 = np.array([[10, 10]]).T

# La réponse retournée doit être $[[20], [100]]$ de dimension (2, 1).
ans = complicated_matrix_function(M_2, a_2, b_2)
print(ans)
print()
print("dimension : ", ans.shape)

only_first_singular_value = get_singular_values(M, 1)
print(only_first_singular_value)


# Maintenant, Récupérons les deux premières valeurs singulières.
# Notez que la première valeur singulière est beaucoup plus grande que la seconde.
first_two_singular_values = get_singular_values(M, 2)
print(first_two_singular_values)

# Assurons-nous que la première valeur singulière dans les deux appels est la même.
assert only_first_singular_value[0] == first_two_singular_values[0]

val, vec = get_eigen_values_and_vectors(N, 1)
print("Première valeur propre = ", val)
print()
print("Premier vecteur propre = \n", vec)
print()
assert len(val) == 1

# Maintenant, récupérons les deux premières valeurs propres et vecteurs propres.
# Votre résultat doit retourner une liste de deux valeurs propres et une liste de deux tableaux (deux vecteurs propres).
val, vec = get_eigen_values_and_vectors(N, 2)
print("Valeurs propres = ", val)
print()
print("Vecteurs propres = \n", vec)
assert len(val) == 2