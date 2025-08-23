import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image
import os
import networkx as nx
from torch_geometric.data import Data
import random
from tqdm import tqdm

def loadData(nodeFeaturePath, edgePath, edgeFeaturePath, nodeLabelPath):
    nodeFeatures = []
    with open(nodeFeaturePath, 'r') as f:
        for line in f:
            nodeFeatures.append([float(x) for x in line.strip().split()[1:]])
    x = torch.tensor(nodeFeatures, dtype=torch.float)
    
    edge_index = []
    with open(edgePath, 'r') as f:
        for line in f:
            edge_index.append([int(x) for x in line.strip().split()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
# edge Fatures
    edge_attr = []
    with open(edgeFeaturePath, 'r') as f:
        for line in f:
            edge_attr.append([float(x) for x in line.strip().split()[2:]])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
# Node labels
    y = []
    with open(nodeLabelPath, 'r') as f:
        for line in f:
            y.append(int(line.strip().split()[1]))
    y = torch.tensor(y, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    print(data)
    return data
