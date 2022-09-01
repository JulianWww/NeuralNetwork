from .tensor import Tensor
import numpy as np

class DenseProcess:
    def __init__(self, inp, layer):
        self.layer = layer
        self.inp = inp
    
    def forward(self):
        val = np.dot(self.layer.w.arr, self.inp.arr) + self.layer.b.arr
        return Tensor(val, self)
    
    def _backup(self, delta, lr):
        self.layer.b.delta = delta
        self.layer.w.delta = np.sum(np.dot(delta.reshape((delta.size, 1)), self.inp.arr.reshape(1, self.inp.arr.size)), axis=0)
        self.inp.delta = np.dot(self.layer.w.arr.T, delta)

        self.layer.w._backup(lr)
        self.layer.b._backup(lr)
        self.inp._backup(lr)

class Dense:
    def __init__(self, input, output):
        self.w = Tensor(np.random.rand(output, input))
        self.b = Tensor(np.random.rand(output))
    
    def forward(self, inp):
        process = DenseProcess(inp, self)
        return process.forward()
    

