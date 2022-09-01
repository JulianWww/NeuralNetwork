import pynet
import numpy as np
from random import randint


class Model:
    def __init__(self):
        self.lin1 = pynet.Dense(2, 1)
    
    def forward(self, x):
        out = self.lin1.forward(x)
        return pynet.sigmoid(out)
    
    def train(self, x, y):
        out = self.forward(x)
        loss = pynet.MSE(out, y)
        loss.backup(0.1)

model = Model()       

x = pynet.Constant (np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]))
y = pynet.Constant (np.array([
    [0.0],
    [1.0],
    [1,0],
    [0.0]
]))


out = model.forward(x[0])
print(out.arr)
out = model.forward(x[1])
print(out.arr)
out = model.forward(x[2])
print(out.arr)
out = model.forward(x[3])
print(out.arr)
print("")

for i in range(100):
    model.train(x[0], y[0])
    model.train(x[1], y[1])
    model.train(x[2], y[2])
    model.train(x[3], y[3])

out = model.forward(x[0])
print(out.arr)
out = model.forward(x[1])
print(out.arr)
out = model.forward(x[2])
print(out.arr)
out = model.forward(x[3])
print(out.arr)