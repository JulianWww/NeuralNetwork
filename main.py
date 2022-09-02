import pynet
import numpy as np
from random import randint


class Model:
    def __init__(self):
        self.lin1 = pynet.Dense(2, 16)
        self.lin2 = pynet.Dense(16, 1)
    
    def forward(self, x):
        out = self.lin1.forward(x)
        out = pynet.leakyReLU(out)
        out = self.lin2.forward(out)
        return pynet.sigmoid(out)
    
    def train(self, x, y):
        out = self.forward(x)
        loss = pynet.MSE(out, y)
        loss.backup(0.1)

model = Model()       

x = pynet.Tensor (np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]))
y = pynet.Constant (np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]))


out = model.forward(x)
print(out.arr)

for i in range(10000):
    model.train(x, y)

out = model.forward(x)
print(out.arr)