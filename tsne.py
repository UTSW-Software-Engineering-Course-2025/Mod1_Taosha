# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os, sys
from adjustbeta import *

def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    X = X - X.mean(axis=0)[None, :]
    X = torch.tensor(X, dtype=torch.float32).to(device)
    _, M = torch.linalg.eig(torch.matmul(X.T, X))
    Y = torch.real(torch.matmul(X, M[:, :no_dims]))
    Y = Y.cpu().detach().numpy()
    return Y

def norm_dist_matrix(P):
    """
    Normalize distance matrix P: set diagnal values to 0 and normalize the matrix to sum to 1.

    Parameters
    ----------
    P : torch.float32 (n,n)
        data input array

    Returns
    -------
    P : torch.float32 (n, n)
        normalized matrix of input P 
    """
    P[torch.arange(P.shape[0]), torch.arange(P.shape[0])] = 0 # forcing diagnal values to be 0
    P = P / torch.sum(P)  # normalizing to sum to 1
    return P


def tsne(X, no_dims_keep=2, perplexity=30, 
         init_momen=0.5, final_momen=0.8, eta=500, min_gain=0.01, T=1000):
    """
    Dimensional reduction using T-SNE.

    Parameters
    ----------
    X : numpy.ndarray (n,d)
        data input array

    no_dims_keep : int 
        target dimensions, default: 2
    
    perplexity: float
        size of neighbourhood, default: 30
    
    init_momen: float
        momentum for early-stage iterations, default: 0.5

    final_momen: float
        momentum for late-stage iterations, default: 0.8

    eta: float
        learning rate, default: 500

    min_gain: float
        minimal gain for each time step, default: 0.2

    T: int
        iteration times, default: 1000

    Returns
    -------
    Y : numpy.ndarray (n, no_dims_keep)
        dimensionally reduced matrix of input X 
    """

    (n, d) = X.shape

    # P: pairwise affinities
    P_cond, beta = adjustbeta(X, perplexity=perplexity)
    P = (P_cond + P_cond.T) / 2
    P = torch.tensor(P, dtype=torch.float32).to(device)
    P = norm_dist_matrix(P)

    # early exaggerate
    P = P * 4
    P[P < 1e-12] = 1e-12

    # initiate
    Y = X[:, :no_dims_keep]
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    delta_Y = torch.zeros((n, no_dims_keep)).to(device)
    gains = torch.ones((n, no_dims_keep)).to(device)
    init_momen = torch.tensor(init_momen, dtype=torch.float32).to(device)
    final_momen = torch.tensor(final_momen, dtype=torch.float32).to(device)
    eta = torch.tensor(eta, dtype=torch.float32).to(device)

    for t in trange(1, T, 1):
        # Q: prob distribution based on Y
        D_Y = calculate_euc_sqr(Y)
        Q_sum = torch.sum(1 / 1 + D_Y)
        Q = 1 / (1 + D_Y) / Q_sum
        Q = norm_dist_matrix(Q)
        Q[Q < 1e-12] = 1e-12

        # dY: gradient of the loss func w.r.t. Y
        _dY_div = (P - Q) / (1 + D_Y)
        _dY_diff = Y[:, None, :] - Y[None, :, :]
        dY = torch.sum(_dY_div[:, :, None] * _dY_diff, axis=1)

        if t < 20:
            momentum = init_momen
        else:
            momentum = final_momen
        gains = (gains + 0.2) * ((dY > 0) != (delta_Y > 0)) + (gains * 0.8) * (
            (dY > 0) == (delta_Y > 0)
        )
        gains[gains < min_gain] = min_gain
        delta_Y = momentum * delta_Y - eta * (gains * dY)
        Y = Y + delta_Y
        if t == 100:
            P = P / 4
        # if t==1:
        #     print(dY)
    Y = Y.cpu().detach().numpy()
    return Y


# if __name__ == "__main__":
#     os.chdir('/Users/gaw/utsw/software_engineering/Module_1_materials/day1/tsne_practice')
#     print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
#     print("Running example on 2,500 MNIST digits...")
#     X = np.loadtxt("mnist2500/mnist2500_X.txt")
#     X = pca(X, 50)
#     labels = np.loadtxt("mnist2500/mnist2500_labels.txt")
#     Y = tsne(X)
#     plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
#     plt.savefig("mnist2500/mnist_tsne.png")


# %%
