# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    I = torch.eye(n).to(device)
    G = kneighbors_graph(X, n_neighbors)
    G = (G + G.T)/2
    L = np.array(laplacian(G).todense())

    L = torch.tensor(L, dtype=torch.float32).to(device)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    lambda_ = torch.tensor(lambda_, dtype=torch.float32).to(device)
    Z = torch.matmul(torch.linalg.inv(I+lambda_*L), X)
    print(Z.shape)
    if no_rotation==False:
        W = torch.linalg.eig(torch.matmul(X.T, Z))[1][:, :d].real
        print(W.shape)
        Z = torch.matmul(Z, W)

    Z = Z.cpu().detach().numpy()
    return Z
