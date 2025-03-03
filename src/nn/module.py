class Module:
    """
    Базовый класс для всех слоев.
    """

    def __call__(self, *args, **kwargs):
        """
        Вызывает метод forward слоя.
        """
        return self.forward(*args, **kwargs)

    def _compute_gradients(self, *args, **kwargs):
        """
        Вызывает метод backward слоя.
        """
        return self.backward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Должен быть переопределён в подклассах.
        """
        raise NotImplementedError("Метод forward должен быть реализован в подклассе")

    def backward(self, *args, **kwargs):
        """
        Должен быть переопределён в подклассах.
        """
        raise NotImplementedError("Метод backward должен быть реализован в подклассе")

    def train(self):
        """
        Должен быть переопределён в подклассах.
        """
        pass

    def eval(self):
        """
        Должен быть переопределён в подклассах.
        """
        pass

    def zero_grad(self):
        """
        Должен быть переопределён в подклассах.
        """
        pass

    def parameters(self):
        return tuple()