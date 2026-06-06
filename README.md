# MLP from Scratch — Custom Autograd Engine

A multi-layer perceptron built completely from scratch without using PyTorch autograd. Includes a custom automatic differentiation engine that supports forward and backward passes through a computation graph.

---

## What's Implemented

- `Value` class with full autograd support
- `Neuron`, `Layer`, `MLP` classes built from scratch
- Forward pass + backpropagation
- Gradient descent training loop
- MSE loss

---

## Results

Trained on a small dataset. Model converged smoothly and predictions closely match targets `[1.0, -1.0, -1.0, 1.0]`

```
Predictions: [0.995, -0.994, -0.988, 0.888]
Targets:     [1.0,   -1.0,  -1.0,   1.0  ]
```

Loss decreased consistently over 20 iterations confirming backprop is working correctly.

---

## What I Learned

How backpropagation actually works under the hood — not just calling `.backward()` but understanding what happens at every single node in the computation graph. Building this made gradient flow and chain rule click in a way that no tutorial could.

---

## Stack

- Python
- Jupyter Notebook

---

## Reference

[Andrej Karpathy — Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
