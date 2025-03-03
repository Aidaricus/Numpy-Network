import numpy as np

class SGD:
    """
    Атрибуты:
    ---------
    params : iterable
        Итерируемый объект, содержащий параметры модели, которые нужно оптимизировать.
    lr : float, optional, default=3e-4
        Learning rate (скорость обучения).
    weight_decay : float, optional, default=0
        Коэффициент для L2-регуляризации.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        """Обнуляет градиенты всех параметров."""
        for param in self.params:
            if param is not None:
                param.grad.fill(0)

    def step(self):
        """Выполняет один шаг градиентного спуска."""
        for param in self.params:
            if param.grad is not None:
                # L2
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data
                param.data -= self.lr * param.grad
