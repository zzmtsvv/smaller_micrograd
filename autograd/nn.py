from typing import Any
import numpy as np

try:
    import tensor
    import functional as F
except ModuleNotFoundError:
    from autograd import tensor
    from autograd import functional as F


class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def parameters(self):
        params = []

        def _params(node):
            if isinstance(node, tensor.Tensor):
                params.append(node)
            elif hasattr(node, "__dict__"):
                for v in node.__dict__.values():
                    _params(v)
        _params(self)
        return params
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class Linear(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int) -> None:
        scale = 1 / np.sqrt(in_features)

        self.w = tensor.Tensor.uniform(-scale, scale, (in_features, out_features))
        self.b = tensor.Tensor.zeros((1, out_features))
    
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        return (x @ self.w) + self.b


def one_hot(labels: np.ndarray):
    o = np.zeros((labels.size, labels.max() + 1))
    o[np.arange(labels.size), labels] = 1
    return tensor.Tensor(o)


def MSELoss(prediction: tensor.Tensor,
            target: tensor.Tensor):
    return F.sum((prediction - target)**2) / target.shape[0]


def CrossEntropyLoss(prediction: tensor.Tensor,
                     target: tensor.Tensor):
    return -F.sum(target * F.log(prediction)) / target.shape[0]
