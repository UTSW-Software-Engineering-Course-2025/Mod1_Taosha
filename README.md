# Software Engineering Module 1: Dimension Reduction
This is a Python library for TSNE and GraphDR algorithms and data visualization. For more detailed documentations, please see docs/build/html/usage.html.

## Methods
TSNE: Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.11 (2008).

GraphDR: Zhou, Jian, and Olga G. Troyanskaya. "An analytical framework for interpretable and generalizable single-cell data analysis." Nature methods 18.11 (2021): 1317-1321.

## Usage
For library dependencies please refer to requirements.txt with Python >= 3.9. To run the scripts please download this repository and run from command line or as a library. Example usages are available in demo.ipynb.

# Command line
cli.py [-h] --data_name DATA_NAME [--data_path DATA_PATH] [--label_path LABEL_PATH] [--data_process] [--method {tsne,graphdr}] [--plot_dim {2d,3d}] [--plot_savepath PLOT_SAVEPATH] [--dims_keep DIMS_KEEP] [--perplexity PERPLEXITY] [--T T] [--lambda_val LAMBDA_VAL] [--no_rotation] [--n_neighbors N_NEIGHBORS]

# Import as a Python library
TSNE: from tsne import tsne

GraphDR: from GraphDR import graphdr