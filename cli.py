#%%
import argparse
from matplotlib import pyplot as plt
import plotly.express as px
import os, sys
import seaborn as sns

from tsne import pca, tsne
from GraphDR import graphdr
from data_proc import read_data

def plot_scatter(data, labels=None, dimension='2d', savepath='output/scatter.png'):
    """
    Visualization and save the final plot.

    Parameters
    ----------
    data : numpy.ndarray (n,d)
        data input array

    labels : numpy.ndarray (n, 1)
        labels for data points, optional

    dimension: str
        number of dimensions for visualization, '2d' or '3d'. Default: '2d'
    
    savepath: str
        path to save the plot. Default: 'output/scatter.png' 
    """
    dirpath = os.path.dirname(savepath)
    if os.path.exists(dirpath)==False:
        os.makedirs(dirpath)

    x = data[:, 0]
    y = data[:, 1]
    if dimension == '2d':
        sns.scatterplot(x=x, y=y, linewidth = 0, s=5, hue=labels)
        plt.savefig(savepath)
    elif dimension == '3d':
        z = data[:, 2]
        fig = px.scatter_3d(x=x, y=y, z=z, color=labels, size=5)
        fig.write_image(savepath)

def wrap(data_name, data_path=None, label_path=None, process=True, method='graphdr', 
         plot_dim='2d', plot_savepath='output/scatter.png', 
         dims_keep = 2, perplexity = 30, 
         lambda_ = 10, no_rotation = True, n_neighbors = 10):
    """
    Wrapping function for command line tools.

    Parameters
    ----------
    **kwargs: parameters for data processing, visualization and tsne/graphdr function.
    """
    print('Reading data...')
    data, labels = read_data(name=data_name, data_path=data_path, label_path=label_path, process=process)
    print('Calculating new matrix...')
    if method == 'tsne':
        Y = tsne(data, no_dims_keep=dims_keep, perplexity=perplexity)
    elif method == 'graphdr':
        # Y = graphdr(data, lambda_=1, no_rotation=True, n_neighbors=10, d=2)
        Y = graphdr(data, lambda_=lambda_, no_rotation=no_rotation, n_neighbors=n_neighbors)
    else:
        ValueError('Method must be graphdr or tsne.')
    print('Plotting...')
    plot_scatter(data=Y, labels=labels, dimension=plot_dim, savepath=plot_savepath)
    print('Done!')

# wrap(data_name='mnist2500', method='tsne')

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run dimension reduction on given dataset and plot results.")
    parser.add_argument('--data_name', required=True, type=str, help="Name of dataset: 'hochgerner', 'mnist2500', or 'customized'.")
    parser.add_argument('--data_path', type=str, default=None, help="Path to custom data file. Required if data_name='customized'.")
    parser.add_argument('--label_path', type=str, default=None, help="Path to label file. Optional.")
    parser.add_argument('--data_process', action='store_false', help="If set, normalize/process data.")
    parser.add_argument('--method', type=str, default='graphdr', choices=['tsne', 'graphdr'], help="Which method to use. Default: graphdr.")
    parser.add_argument('--plot_dim', type=str, default='2d', choices=['2d', '3d'], help="Plot dimension. Default: 2d.")
    parser.add_argument('--plot_savepath', type=str, default='output/scatter.png', help="Where to save the plot. Default: output/scatter.png.")
    parser.add_argument('--dims_keep', type=int, default=2, help="Number of dimensions to keep. Default: 2.")
    parser.add_argument('--perplexity', type=float, default=30.0, help="Perplexity for t-SNE. Default: 30.0.")
    parser.add_argument('--lambda_val', type=float, default=10.0, help="Regularization strength for GraphDR. Default: 10.0.")
    parser.add_argument('--no_rotation', action='store_false', help="If set, disables rotation in GraphDR. Default: True.")
    parser.add_argument('--n_neighbors', type=int, default=10, help="Number of neighbors for GraphDR. Default: 10.")
    args = parser.parse_args()

    data_name = args.data_name
    data_path = args.data_path
    label_path = args.label_path
    data_process = args.data_process
    method = args.method
    plot_dim = args.plot_dim
    plot_savepath = args.plot_savepath
    dims_keep = args.dims_keep
    perplexity = args.perplexity
    lambda_val = args.lambda_val
    no_rotation = args.no_rotation
    n_neighbors = args.n_neighbors

    wrap(data_name=data_name, data_path=data_path, label_path=label_path, process=True, method=method, 
         plot_dim='2d', plot_savepath='output/scatter.png', 
         dims_keep = dims_keep, perplexity = perplexity, 
         lambda_ = lambda_val, no_rotation = no_rotation, n_neighbors = n_neighbors)