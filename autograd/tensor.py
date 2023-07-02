from typing import Union, List, Tuple, Optional
import numpy as np
from collections import deque

try:
    import functional as F
except ModuleNotFoundError:
    from . import functional as F


Scalar = Union[int, float]
_Shape = Tuple[int, ...]


class Tensor:
    def __init__(self,
                 data: Union[Scalar, List[Scalar]]) -> None:
        if isinstance(data, (int, float)):
            data = [data]
        
        self.data = np.array(data)

        self.grad = None
        self.children = []
    
    @staticmethod
    def sort(root):
        sort_, visited = deque(), set()

        def dfs(node):
            for child in node.children:
                if child not in visited:
                    visited.add(child)
                    dfs(child)
            sort_.appendleft(node)
        
        dfs(root)
        return sort_

    @staticmethod
    def ones(shape: _Shape):
        return Tensor(np.ones(shape))
    
    @staticmethod
    def ones_like(object):
        return Tensor(np.ones_like(object))

    @staticmethod
    def zeros(shape: _Shape):
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def zeros_like(object):
        return Tensor(np.zeros_like(object))

    @staticmethod
    def uniform(low=0.0, high=1.0, shape: Optional[_Shape] = None):
        return Tensor(np.random.uniform(low=low, high=high, size=shape))
    
    @staticmethod
    def randn(shape: _Shape):
        return Tensor(np.random.randn(*shape))

    @property
    def shape(self) -> _Shape:
        return self.data.shape

    @property
    def nonerror_grad(self):
        return self.zeros(self.shape) if self.grad is None else self.grad

    @staticmethod
    def unbroadcast(out, input_shape: _Shape):
        sum_axis = None

        if input_shape != (1,):
            sum_axis = tuple([
                i for i in range(len(input_shape)) if input_shape[i] == 1 and out.shape[i] > 1
            ])
        return Tensor(out.data.sum(axis=sum_axis).reshape(input_shape))

    @property
    def is_leaf_node(self) -> bool:
        return not bool(self.children)

    def backward(self):
        topsort = self.sort(self)
        self.grad = self.ones(self.shape)

        for root in topsort:
            if not root.is_leaf_node:
                out_grad = root._backward(root.grad.data)
                
                for i, child in enumerate(root.children):
                    child.grad = child.nonerror_grad + out_grad[i]
                    child.grad = self.unbroadcast(child.grad, child.shape)
    
    def reshape(self, *shapes):
        return Tensor(self.data.reshape(*shapes))
    
    def sum(self, axis=None):
        return F.sum(self, axis=axis)
    
    def norm(self):
        return F.norm(self)
    
    def sigmoid(self):
        return F.sigmoid(self)
    
    def softmax(self, axis=None):
        return F.softmax(self, axis=axis)
    
    def relu(self):
        return F.relu(self)
    
    def __matmul__(self, other):
        return F.matmul(self, other)
    
    def __rmatmul__(self, other):
        return F.matmul(other, self)
    
    def __add__(self, other):
        return F.add(self, other)
    
    def __radd__(self, other):
        return F.add(other, self)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __iadd__(self, other):
        other = F.format_input(other)
        self.data = self.data + other.data
        return self
    
    def __isub__(self, other):
        other = F.format_input(other)
        self.data = self.data - other.data

        return self
    
    def __mul__(self, other):
        return F.mul(self, other)
    
    def __rmul__(self, other):
        return F.mul(other, self)
    
    def __pow__(self, other):
        return F.pow(self, other)
    
    def __rpow__(self, other):
        return F.pow(other, self)
    
    def __truediv__(self, other):
        return F.div(self, other)
    
    def __rtruediv__(self, other):
        return F.div(other, self)
    
    def __neg__(self):
        return F.mul(self, -1)
    
    def __repr__(self):
        repr_ = "\n".join([str(line) for line in self.data])
        return f"Tensor({repr_})"
