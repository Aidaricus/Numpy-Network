import numpy as np

class Parameter:
    """
    Класс для хранения параметров модели и их градиентов.

    Параметры:
    ----------
    shape: tuple or int
        Определяет размер массива параметров.

    Атрибуты:
    ---------
    data: np.ndarray
        Массив параметров модели.
    grad: np.ndarray
        Градиенты параметров (обнуляются перед обучением).
    m: np.ndarray or None
        Переменная первого момента (используется в оптимизаторах, например, Adam).
    v: np.ndarray or None
        Переменная второго момента (используется в оптимизаторах, например, Adam).
    """

    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape)
        self.grad = np.zeros(shape)
        self.m = None
        self.v = None

    def _init_params(self, method='kaiming'):
        """
        Инициализация параметров модели.

        Параметры:
        ----------
        method: str, по умолчанию 'kaiming'
            Метод инициализации параметров. Доступные методы:
            - 'kaiming': Инициализация Kaiming He.
            - 'zeros': Инициализация нулями.
            - 'ones': Инициализация единицами.

        Исключения:
        -----------
        ValueError
            Если указан неизвестный метод инициализации.
        """
        if method == 'kaiming':
            fan_in = self.shape[0] if isinstance(self.shape, tuple) else self.shape
            self.data = np.random.randn(*self.shape) * np.sqrt(2 / fan_in)
        elif method == 'zeros':
            self.data = np.zeros_like(self.data)
        elif method == 'ones':
            self.data = np.ones_like(self.data)
        else:
            raise ValueError(f"Неизвестный метод инициализации: {method}")
        return self