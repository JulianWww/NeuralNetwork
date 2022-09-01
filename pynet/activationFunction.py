from .tensor import Tensor
import numpy as np

class ActivationFunction:
    def __init__(self, x):
        self.x = x
    
    def _backup(self, delta, lr):
        self.x.delta = self._derivative() * delta
        self.x._backup(lr)

class _LeakyReLU(ActivationFunction):
    def __init__(self, x, a=0.3):
        super(_LeakyReLU, self).__init__(x)
        self.a = a

    def _derivative(self):
        deriv = np.ones_like(self.x.arr)
        deriv[np.where(self.x.arr<0)] = 1/self.a
        return deriv

def leakyReLU(x, a=0.3):
    arr = np.copy(x.arr)
    mask = np.where(arr<0)
    arr[mask] *= a
    return Tensor(arr, _LeakyReLU(x, a))



class _Sigmoid(ActivationFunction):
    def _derivative(self):
        return (np.e**(-self.x.arr))/((1+np.e**(-self.x.arr))**2)

def sigmoid(x):
    return Tensor(1/(1+np.e**(-x.arr)), _Sigmoid(x))

