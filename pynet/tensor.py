import numpy as np

class Tensor:
    def __init__(self, arr, function=None):
        self.arr = arr
        self.function = function
        self.delta = None
    
    def __str__(self):
        return f"{str(self.arr)}"
    
    def backup(self, value, lr):
        self.delta = self.arr - value.arr
        self._backup(lr)

    def _backup(self, lr):
        if not self.function is None:
            self.function._backup(self.delta, lr)
        self.update(lr)
    
    def update(self, lr):
        self.arr -= self.delta * lr

    __repr__=__str__

    def __getitem__(self, *args):
        return Tensor(self.arr.__getitem__(*args))

class Constant(Tensor):
    def __init__(self, arr, function=None):
        super(Constant, self).__init__(arr, function)

    def update(self, _):
        pass