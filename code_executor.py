#%%
import numpy as np
from kmeansPro import kmeansPro as kmp
# Model further settings
np.set_printoptions(precision=4)

# data creation
Num_sam=100
Num_var=2
num_clusters=3
datapoints=np.random.rand(Num_sam,Num_var)
new_candidates=np.random.rand(1,Num_var)

# %%
# Training the model
data_clustered=kmp()
data_clustered.fit(datapoints,Num_repeat=20)
# %%
#Trained Model Information
print(f'The best suggested Number of Cluster={data_clustered.NumCluster}')
print(f'goodness of the trained model = {data_clustered.goodness}')
print(f'clusters counting memebrs{data_clustered.clustr_counts}')

# %%
# Clusters Visualization (including new candidate)
data_clustered.visual_ploting(None,new_candidates,method=1) #assigning to the nearest center
data_clustered.visual_ploting(None,new_candidates,method=2,K_nn=3) #assigning based on nearest neighbors

