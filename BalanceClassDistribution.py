import torch
from torch_geometric.utils import subgraph
from collections import Counter

# we define this module to balance nmber of Clases
def NumberOfSamplesClass(labels):
    classCounts = Counter(labels.tolist())
    for class_id, count in sorted(classCounts.items()):
        print(f"Class {class_id}: {count} samples")
    return classCounts

def AdjustClassSamples(data):
    print("Class distribution before adjusting the outlier:")
    initial_counts = NumberOfSamplesClass(data.y) #count_samples_per_class(data.y)
# sho the outlier class
    classCounts = torch.bincount(data.y)
    outlierClass = torch.argmax(classCounts).item()
    print(f"\nOutlier class ID: {outlierClass}") 
# Calculate the mean number of samples per class by excluding the outlier clas
    meanSampleCount = int(classCounts[classCounts != classCounts[outlierClass]].float().mean().item())
    print(f"Mean sample Cont: {meanSampleCount}") 
    outlierIndices = (data.y == outlierClass).nonzero(as_tuple=True)[0] # Indice of outlier class
# Select only number of men sampls from the outlier class
    if len(outlierIndices) > meanSampleCount:
        selectedOutlierIndices = outlierIndices[:meanSampleCount]
    else:
        selectedOutlierIndices = outlierIndices
    
    keepMask = torch.ones(data.num_nodes, dtype=torch.bool)
    keepMask[outlierIndices] = False  # Temporaly excluding all outlier samples
    keepMask[selectedOutlierIndices] = True  # Include only selected outlier samples
# Apply these maks
    data.x = data.x[keepMask]
    data.y = data.y[keepMask]
# Adjust index` to retain only edges that connect nodes in `keepMask`
    data.edge_index, data.edge_attr = subgraph(keepMask, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = data.x.size(0)
    print("\nClass ditribution after balancement of number clases:")
    adjustedCounts = NumberOfSamplesClass(data.y)
    
    print(f"\nOutlier class samle count adjusted to {meanSampleCount}. Other classes remain unhanged.")
    print(f"New total number of nodes: {data.num_nodes}")
    return data
