from .tensor import Tensor
import numpy as np

class DenseProcess:
    def __init__(self, inp, layer):
        self.layer = layer
        self.inp = inp
    
    def forward(self):
        val = np.dot(self.inp.arr, self.layer.w.arr) + self.layer.b.arr
        return Tensor(val, self)
    
    def _backup(self, delta, lr):
        self.layer.b.delta = np.sum(delta, axis=0)
        delta = np.expand_dims(delta, axis=1)
        out = np.sum(np.matmul(
            np.expand_dims(self.inp.arr, axis=2),
            delta
        ), axis=0)
        self.layer.w.delta = out
        self.inp.delta = np.dot(delta, self.layer.w.arr.T)[:,0]   

    
        self.layer.w._backup(lr)
        self.layer.b._backup(lr)
        self.inp._backup(lr)

class Dense:
    def __init__(self, input, output):
        self.w = Tensor(2*np.random.rand(input, output)-1)
        self.b = Tensor(2*np.random.rand(1, output)-1)
    
    def forward(self, inp):
        process = DenseProcess(inp, self)
        return process.forward()
    

