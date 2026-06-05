import math
class Value():
    #Parameter of the object
    def __init__(self,data,_children=(),grad=0,label='',_op=''):
        self.data = data
        self._prev = set(_children)
        self.grad = grad
        self._backward = lambda :None
        self.label = label
        self._op = _op
    
    #Adding function 
    def __add__(self,other):
        if(isinstance(other,(int,float))):  # If other is int then convert to other eg.. Value*2
            other = Value(other)
        out = Value(self.data+other.data,(self,other),label='add',_op='+')

        #Calculate the gradient using calculus
        def _backward():             # // f(x+h)-f(x)/h  : This is to help to Derivative
            self.grad += out.grad # +out.grad This is from the chan rule
            other.grad += out.grad
        out._backward = _backward;
        return out 
    
    #Subration function 
    def __sub__(self,other):
        if(isinstance(other,(int,float))):
            other = Value(other)
        out = Value(self.data-other.data,(self,other),label='sub',_op='-')

        #Calculate the gradient using calculus
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad
        out._backward = _backward; 
        return out
    
    #Multiplication function
    def __mul__(self,other):
        if(isinstance(other,(int,float))):
            other = Value(other)
        out = Value(self.data*other.data,(self,other),label='mul',_op='*')

        #Calculate the gradient using calculus
        def _backward():
            self.grad += other.data * out.grad # here if we calcaute the gradient of the out with respect to self then other will be constant so it's other.data* 
            other.grad += self.data * out.grad
        out._backward = _backward; 
        return out

    #Power function
    def __pow__(self,other):
        out = Value(self.data**other.data,label='pow',_op='**')

        def _backward():
            self.grad = (other.data)*(self.data**(other.data-1))*out.grad 
            other.grad = (self.data)*(other.data**(self.data-1))*out.grad 
        out._backward = _backward
        return out 

    #If function if fails
    #right hand operations 
    def __rmul__(self,other): # from 2*Value -> Value*2
        return self*other;
    def __radd__(self,other):
        return self+other
    def __rsub__(self,other):
        return self-other
    def __rpow__(self,other):
        return self**other

    #Activation functions
    #tanh
    def tanh(self):
        if(isinstance(self,(int,float))):
            self = Value(self)
        t = self.data  
        out = Value((math.exp(2*t)-1)/(math.exp(2*t)+1),label='tanh'); #This is the hyprbolic function 

        def _backward():
            self.grad += (1-t**2) * out.grad#(1-t^2) //This is the derivative of the tanh

        out._backward = _backward

        return out
    #Display or Print function
    def __repr__(self): 
        return f"Value({self.data}:{self.grad})"

    #topo sort 
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()



