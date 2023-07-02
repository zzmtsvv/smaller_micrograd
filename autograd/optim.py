class SGD:
    def __init__(self,
                 parameters,
                 lr: float = 3e-4,
                 normalize: bool = False) -> None:
        self.parameters = parameters
        self.lr = lr
        self.normalize = normalize
    
    def step(self):
        for param in self.parameters:
            if self.normalize:
                step = param.grad / param.grad.norm()
            else:
                step = param.grad
            
            param -= self.lr * step
