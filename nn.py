import random
from engine import Value 
#Neuron or Perceptron 
#There are the main component in a neural network 
'''
overview of how the simple neuron work
neuron output = input*weights + bias
next we calcuate the loss (mle,mse) 
eg.. Mean squared error 

MSE(loss function) = (predictedValue-ActuralValue)^2

then we iterate to reduce this loss 
weights = weights+learningRate*gradient

'''
class Neuron():
    def __init__(self,nip): # size of the input layer
        #Random weights
        self.w =[Value(random.uniform(-1,1)) for _ in range(nip)] # This create the random weights between -1 to 1 
        self.b = Value(random.uniform(-1,1)) # we are using just one bias
     
    def __call__(self,x): #when the object is called this function is called automatically // x is the input

        out  = sum([i*w for i,w in zip(self.w,x)],self.b) 
        act = out.tanh()
        return act
    def parameters(self):
        return self.w+[self.b]


class Layer():
    def __init__(self,nip,nout):
       self.neuron= [Neuron(nip) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neuron]
        return out[0] if(len(out)==1)  else out
    def parameters(self):
        return [p for neuron in self.neuron for p in neuron.parameters() ]

class MLP():
    def __init__(self,nip,nout):
        zs = [nip]+nout
        self.layers = [Layer(zs[i],zs[i+1]) for i in range(len(nout))]
    def __call__(self,x):
        for layers in self.layers:
            x = layers(x)
        return x
    def parameters(self):
        return [p for layers in self.layers for p in layers.parameters()]



