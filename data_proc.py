import os, sys
import pickle
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_data(name='customized', data=None, labels=None, process=True,
              data_path = None, label_path = None):
    """
    Wrapper for data processing.

    Parameters
    ----------
    name : str
        Name of dataset: 'hochgerner', 'mnist2500', or 'customized'

    data : numpy.ndarray (n,d)
        Data array, required if name = customized and data_path = None

    labels : numpy.ndarray (n, 1)
        Labels for data points

    process: bool
        Whether to process/normalize the data. default: True
    
    data_path: path
        required if name = customized and data_path = False. But be a numpy array (samples, features)
    
    label_path: path

    Returns
    -------
    data : numpy.ndarray (samples, features)
        feature matrix for training

    labels: numpy.ndarray (samples, 1)
        Labels for data points
    """
    
    if name == 'mnist2500':
        data = load_pkl('datasets/mnist2500/pca_data.pkl')
        labels = np.loadtxt("datasets/mnist2500/mnist2500_labels.txt")
        labels = labels.astype(str)
    elif name == 'hochgerner':
        data = load_pkl('datasets/hochgerner/pca_data.pkl')
        labels = pd.read_csv('datasets/hochgerner/hochgerner_2018.anno',sep='\t',header=None)[1].values
    elif name == 'customized':
        if data_path != None:
            data = np.load(data_path)
        if label_path != None:
            labels = np.load(data_path)
        if process == True:
            #We will first normalize each cell by total count per cell.
            percell_sum = data.sum(axis=0)
            pergene_sum = data.sum(axis=1)

            preprocessed_data = data / percell_sum.values[None, :] * np.median(percell_sum)
            preprocessed_data = preprocessed_data.values

            #transform the preprocessed_data array by `x := log (1+x)`
            preprocessed_data = np.log(1 + preprocessed_data)

            #standard scaling
            preprocessed_data_mean = preprocessed_data.mean(axis=1)
            preprocessed_data_std = preprocessed_data.std(axis=1)
            preprocessed_data = (preprocessed_data - preprocessed_data_mean[:, None]) / \
                                preprocessed_data_std[:, None]
            
            pca = PCA(n_components = 50)
            pca.fit(preprocessed_data.T)
            data = pca.transform(preprocessed_data.T)
    else:
        ValueError('Name must be mnist2500, hochgerner or customized!')
    return data, labels


    