import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class Dilator:
    def __init__(self, device):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                            stride=1, padding=1, bias=False, device=device, dtype=torch.float32)
        self.conv.weight = nn.Parameter(torch.Tensor([[[[1,1,1],[1,1,1],[1,1,1]]]], device=device), requires_grad=False)
    
    def __call__(self, image):
        x = self.conv(image)
        return x > 0

class NodesDeg4Detector:
    def __init__(self, device):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, 
                            stride=1, padding=1, bias=False, device=device, dtype=torch.float32)
        self.conv.weight = nn.Parameter(torch.Tensor([[[[1,1],[1,1]]]], device=device), requires_grad=False)
    
    def __call__(self, image):
        x = self.conv(image)
        x = (x == 4)
        return x[:,:-1,:-1]
    
class NodesDeg3Detector:
    def __init__(self, device):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                            stride=1, padding=1, bias=False, device=device, dtype=torch.float32)
        self.conv.weight = nn.Parameter(torch.Tensor([[[[1,1,1],[1,0,1],[1,1,1]]]], device=device), requires_grad=False)
    
    def __call__(self, image):
        x = self.conv(image)
        x = (x >= 3)
        x = torch.logical_and(image, x)
        return x

class NodesDeg1Detector:
    def __init__(self, device):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                            stride=1, padding=1, bias=False, device=device, dtype=torch.float32)
        self.conv.weight = nn.Parameter(torch.Tensor([[[[1,1,1],[1,0,1],[1,1,1]]]], device=device), requires_grad=False)
    
    def __call__(self, image):
        x = self.conv(image)
        x = (x == 1)
        x = torch.logical_and(image, x)
        return x

def get_segmented_skeleton(image: torch.tensor, device='cpu'):
    det4 = NodesDeg4Detector(device)
    det3 = NodesDeg3Detector(device)
    det1 = NodesDeg1Detector(device)

    crossings    = det4(image)
    bifurcations = det3(image)
    endpoints    = det1(image)
    
    all_nodes = torch.any(torch.stack([crossings, bifurcations, endpoints]), dim=0)
    dilate = Dilator(device)
    segmented_skeleton = image.clone().detach()
    segmented_skeleton[dilate(all_nodes.float())] = 0

    nodes_crossings = np.stack(np.where(crossings.squeeze()), axis=-1)
    nodes_bifurcations = np.stack(np.where(bifurcations.squeeze()), axis=-1)
    nodes_endpoints = np.stack(np.where(endpoints.squeeze()), axis=-1)
    nodes = np.vstack([
        np.hstack([nodes_crossings,    np.full((nodes_crossings.shape[0],    1), 4)]),
        np.hstack([nodes_bifurcations, np.full((nodes_bifurcations.shape[0], 1), 3)]),
        np.hstack([nodes_endpoints,    np.full((nodes_endpoints.shape[0]   , 1), 1)])
    ])
    
    return nodes, segmented_skeleton

def plot_nodes_masks(ax, endpoints=None, bifurcations=None, crossings=None):
    if endpoints is not None:
        y_deg1, x_deg1 = np.where(endpoints)
        ax.scatter(x=x_deg1, y=y_deg1, marker='o', c='b', s=3)
    
    if bifurcations is not None:
        y_deg3, x_deg3 = np.where(bifurcations)
        ax.scatter(x=x_deg3, y=y_deg3, marker='^', c='r', s=3)
    
    if crossings is not None:
        y_deg4, x_deg4 = np.where(crossings)
        ax.scatter(x=x_deg4, y=y_deg4, marker='^', c='r', s=3)