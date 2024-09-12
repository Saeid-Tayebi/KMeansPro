# Robust K-Means Clustering Algorithm

## Overview

This project introduces a robust k-means clustering algorithm that addresses the variability of traditional k-means due to random initialization. The algorithm ensures more consistent and reliable cluster assignments by using a refined approach to initialization and iteration.

Key features of this implementation include:
- Automatic calculation of the maximum possible number of clusters, ensuring no cluster contains less than a specified threshold of the dataset.
- Methods for assigning new observations to clusters using either nearest center or nearest neighbor.
- Visualization tools to display data distribution and new data point assignments.

## Features

### 1. `max_possible_num_cluster(self, data: np.ndarray, threshold=0.2, K0=None)`

This method finds the maximum possible number of clusters, ensuring each cluster contains at least `threshold` proportion of the dataset. If `K0` is provided, the method checks its validity; otherwise, it iteratively decreases `K0` to find an appropriate number of clusters.

**Parameters:**
- `data`: Input data for clustering (numpy array).
- `threshold`: Minimum portion of the dataset each cluster should contain (default is 0.2).
- `K0`: Initial guess for the number of clusters (optional).

### 2. `fit(self, data: np.ndarray, k=None, Num_repeat=50)`

Fits the k-means model to the data. If `k` is not provided, the number of clusters is calculated using the `max_possible_num_cluster` method. The algorithm runs for `Num_repeat` iterations to identify the best cluster assignments, minimizing variability and optimizing the distribution between samples and cluster centers.

**Parameters:**
- `data`: Input data for clustering (numpy array).
- `k`: Number of clusters (optional).
- `Num_repeat`: Number of iterations to repeat the clustering process (default is 50).

### 3. `hosting_cluster(self, new_candidates, method=1, K_nn=3)`

Assigns new data points to clusters based on two methods:
- **Method 1:** Assigns new data points based on the nearest cluster center.
- **Method 2:** Assigns new data points based on the K nearest neighbors within the clusters.

**Parameters:**
- `new_candidates`: New data points to assign (numpy array).
- `method`: Method for determining the hosting cluster (1 for nearest center, 2 for nearest neighbors).
- `K_nn`: Number of nearest neighbors to consider when using method 2 (default is 3).

### 4. `visual_plotting(self, axis_plot=None, new_candidates=None)`

Visualizes the clusters in the latent space and shows the new data points with their assigned clusters. This helps in understanding the distribution of data points and cluster boundaries.

**Parameters:**
- `axis_plot`: Axis for plotting (optional).
- `new_candidates`: New data points to visualize along with the clusters (optional).

## Usage

Here is an example of how to use the robust k-means clustering algorithm in Python:

```python
# %%
import numpy as np
from kmeansPro import kmeansPro as kmp

# Model further settings
np.set_printoptions(precision=4)

# Data creation
Num_sam = 100
Num_var = 2
datapoints = np.random.rand(Num_sam, Num_var)
new_candidates = np.random.rand(1, Num_var)

# %%
# Training the model
data_clustered = kmp()
data_clustered.fit(datapoints, Num_repeat=20)

# %%
# Trained Model Information
print(f'The best suggested Number of Clusters = {data_clustered.NumCluster}')
print(f'Goodness of the trained model = {data_clustered.goodness}')
print(f'Cluster member counts: {data_clustered.clustr_counts}')

# %%
# Clusters Visualization (including new candidate)
data_clustered.visual_plotting(None, new_candidates, method=1)  # Assigning to the nearest center
data_clustered.visual_plotting(None, new_candidates, method=2, K_nn=3)  # Assigning based on nearest neighbors
