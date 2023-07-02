from __future__ import annotations
from typing import Any
import numpy as np
from scipy.special import expit

try:
    import tensor
except ModuleNotFoundError:
    from . import tensor


def format_input(data):
    if isinstance(data, tensor.Tensor):
        return data
    return tensor.Tensor(data)


def softmax(x, axis=None):
    exp = np.e ** x
    return exp / exp.sum(axis=axis)


class Ctx:
    def __init__(self) -> None:
        self.saved = []
    
    def save_for_backward(self, *args):
        self.saved.extend(args)


class Function:
    @staticmethod
    def forward(ctx: Ctx, *args, **kwargs):
        raise NotImplementedError()
    
    @staticmethod
    def backward(ctx: Ctx, *args, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, *args: Any, **kwargs: Any):
        ctx, args = Ctx(), [format_input(arg) for arg in args]

        out = self.forward(ctx, *args, **kwargs)

        out.children = [child for child in args if isinstance(child, tensor.Tensor)]
        out._backward = lambda x: self.backward(ctx, x)

        return out


class Add(Function):
    @staticmethod
    def forward(ctx: Ctx,
                a: tensor.Tensor,
                b: tensor.Tensor) -> tensor.Tensor:
        return tensor.Tensor(a.data + b.data)
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in, grad_in]


class Sum(Function):
    @staticmethod
    def forward(ctx: Ctx, x: tensor.Tensor, axis=None) -> tensor.Tensor:
        ctx.save_for_backward(x, axis)
        return tensor.Tensor(x.data.sum(axis=axis, keepdims=True))
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        return [grad_in]


class Mul(Function):
    @staticmethod
    def forward(ctx: Ctx,
                a: tensor.Tensor,
                b: tensor.Tensor) -> tensor.Tensor:
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data * b.data)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        a, b = ctx.saved
        return [grad_in * b.data, grad_in * a.data]


class Div(Function):
    @staticmethod
    def forward(ctx: Ctx,
                a: tensor.Tensor,
                b: tensor.Tensor) -> tensor.Tensor:
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data / b.data)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        a, b = ctx.saved
        return [grad_in / b.data, grad_in * (-a.data / b.data ** 2)]


class MatMul(Function):
    @staticmethod
    def forward(ctx: Ctx, a, b) -> tensor.Tensor:
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data @ b.data)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        a, b = ctx.saved
        return [grad_in @ b.data.T, a.data.T @ grad_in]


class Pow(Function):
    @staticmethod
    def forward(ctx: Ctx,
                a,
                b) -> tensor.Tensor:
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data ** b.data)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        a, b = ctx.saved

        da = b.data * a.data ** (b.data - 1)
        db = a.data ** b.data * np.log(a.data)

        return [grad_in * da, grad_in * db]


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Ctx, x) -> tensor.Tensor:
        exp = expit(x.data)
        ctx.save_for_backward(exp)
        return tensor.Tensor(exp)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        exp = ctx.saved[0]
        return [grad_in * exp * (1 - exp)]


class ReLU(Function):
    @staticmethod
    def forward(ctx: Ctx, x) -> tensor.Tensor:
        ctx.save_for_backward(x)
        return tensor.Tensor(np.maximum(0, x.data))
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        x = ctx.saved[0]
        return [grad_in * (x.data >= 0)]


class Norm(Function):
    @staticmethod
    def forward(ctx: Ctx, x) -> tensor.Tensor:
        norm = np.linalg.norm(x.data)
        ctx.save_for_backward(x, norm)
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        x, norm = ctx.saved
        return [grad_in * (x.data / norm)]


class Log(Function):
    @staticmethod
    def forward(ctx: Ctx, x) -> tensor.Tensor:
        ctx.save_for_backward(x)
        return tensor.Tensor(np.log(x.data))
    
    @staticmethod
    def backward(ctx: Ctx, grad_in):
        x = ctx.saved[0]
        return [grad_in / x.data]


add, sum, mul, div, matmul = Add(), Sum(), Mul(), Div(), MatMul()
pow, sigmoid, relu, norm, log = Pow(), Sigmoid(), ReLU(), Norm(), Log()
