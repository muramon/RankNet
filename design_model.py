import torch.nn as nn

def design_model(actf, dims):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(actf)
    layers.append(nn.Linear(dims[-1], 1))
    return layers