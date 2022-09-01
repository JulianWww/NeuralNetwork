from .tensor import Tensor
import numpy as np

class MSE:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self):
        assert self.x.arr.shape == self.y.arr.shape, f"MSE Dimension incorrect for x {self.x.arr.shape} and y {self.y.arr.shape}"
        arr = self.x.arr - self.y.arr
        return Tensor(1/(self.x.arr.size) * arr.dot(arr), self)
    
    def derivative(self):
        dx = 2/(self.x.arr.size) * (self.x.arr - self.y.arr)
        return dx, -dx

    def backup(self, lr):
        dx, dy = self.derivative()

        self.x.delta = dx
        self.y.delta = dy

        self.x._backup(lr)
        self.x._backup(lr)
