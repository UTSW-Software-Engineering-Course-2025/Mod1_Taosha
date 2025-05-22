# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph

def graphdr(X, lambda_=10, no_rotation=True, n_neighbors=10, d=2):
    """
    Dimensional reduction using GraphDR.

    Parameters
    ----------
    X : numpy.ndarray (n,d)
        data input array

    lambda_ : float
        regularization strength, default: 10

    no_ratation: bool
        perform dimension reduction or not, if True then no reduction. default: True
    
    n_neighbors: int
        size of neighbourhood, default: 10
    
    d: int
        target dimensions, only necessary when no_rotation is False. default: 2

    Returns
    -------
    Z : numpy.ndarray (n, no_dims_keep)
        dimensionally reduced matrix of input X  
    """
    (n, p) = X.shape

    I = np.eye(N=n)
    G = kneighbors_graph(X, n_neighbors)
    G = (G + G.T)/2
    L = laplacian(G)
    Z = np.dot(np.linalg.inv(I+lambda_*L), X)
    if no_rotation==False:
        W = np.linalg.eig(np.dot(X.T, Z)).eigenvectors[:d]
        Z = np.dot(Z, W)

    return np.array(Z)

# # %%
# import pickle
# with open('hochgerner/pca_data.pkl', 'rb') as f:
#     pca_data = pickle.load(f)
# anno = pd.read_csv('hochgerner/hochgerner_2018.anno',sep='\t',header=None)
# anno = anno[1].values

# graphdr_data  = graphdr(pca_data, lambda_=10, no_rotation=True)

# plt.figure(figsize=(15,10))
# sns.scatterplot(x=graphdr_data[:,0], y=graphdr_data[:,1], linewidth = 0, s=3, hue=anno)
# plt.xlabel('GraphDR 1')
# plt.ylabel('GraphDR 2')
# %%
