import os
import nn
import engine 
from nn import MLP 
from engine import Value 
import torch.nn as nn
import torch.nn.functional as F


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3,[4,1])
for x in xs:
    yp = mlp(x) 
    loss = (yp-ys)**2

loss = sum((pred - y)**2 for pred, y in zip(yp, ys))









