# %%
import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import trange
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Hbeta(D, beta=1.0):
    """
    Compute entropy(H) and probability(P) from nxn distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        distance matrix (n,n)
    beta : float
        precision measure
    .. math:: \beta = \frac{1}/{(2 * \sigma^2)}

    Returns
    -------
    H : float
        entropy
    P : numpy.ndarray
        probability matrix (n,n)
    """
    beta = torch.tensor(beta, dtype=torch.float32).to(device)
    num = torch.exp(-D * beta)
    den = torch.sum(torch.exp(-D * beta), 0)
    num = num.to(device)
    den = den.to(device)
    P = num / den
    H = torch.log(den) + beta * torch.sum(D * num) / (den)
    P = P.cpu().detach().numpy()
    H = H.cpu().detach().numpy()
    return H, P

def calculate_euc_sqr(X):
    """
    Calculate the squared Euclidean distance matrix.

    Parameters
    ----------
    X : numpy.ndarray (n,d)
        data input array

    Returns
    -------
    D : numpy.ndarray (n, n)
        distance matrix of X.
    """
    X = torch.tensor(X, dtype=torch.float32).to(device)
    X_sum = torch.sum(X**2, axis=1, keepdims=True)
    D = X_sum - 2*torch.matmul(X, X.T) + X_sum.T
    return D

def adjustbeta(X, tol=1e-5, perplexity=30):
    """
    Precision(beta) adjustment based on perplexity

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number of neighbors

    Returns
    -------
    P : numpy.ndarray
        probability matrix (n,n)
    beta : numpy.ndarray
        precision array (n,1)
    """
    (n, d) = X.shape
    # Need to compute D here, which is nxn distance matrix of X
    D = calculate_euc_sqr(X)

    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    print('choosing best beta...')
    for i in trange(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    return P, beta


# %%
