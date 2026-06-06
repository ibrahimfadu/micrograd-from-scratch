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

loss=[]

for x,y in zip(xs,ys):
    yp=mlp(x)
    loss += [(yp-y)**2]

loss = sum(loss)

loss.backward()

mlp.parameters()

print(loss)
    








