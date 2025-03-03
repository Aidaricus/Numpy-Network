from src.tensor import Tensor

class Sequential:
    """
    Класс Sequential для определения последовательной структуры нейронной сети.

    Sequential объединяет несколько слоев (модулей) в последовательную модель, 
    где выход одного слоя передается на вход следующего.

    Параметры:
    ----------
    *args : список модулей (слоев)
        Последовательность элементов нейронной сети.

    Исключения:
    -----------
    ValueError
        Если передана пустая последовательность слоев.

    Пример инициализации:
    -----------
    model = Sequential(Layer1, Layer2, Layer3)
    """

    def __init__(self, *modules):
        if len(modules) == 0:
            raise ValueError("В последовательности должен быть хотя бы один элемент")
        self.modules = modules

    def forward(self, x):
        """
        Прямой проход через все слои модели.

        Параметры:
        ----------
        x: np.ndarray
            Входные данные.

        Возвращает:
        -----------
        Tensor
            Выходные данные после прохождения через все слои.
        """
        for module in self.modules:
            x = module(x)
        return Tensor(x, self)

    def __call__(self, x):
        """Позволяет вызывать экземпляр Sequential как функцию."""
        return self.forward(x)

    def parameters(self):
        """
        Генератор, возвращающий параметры всех модулей модели.

        Возвращает:
        -----------
        Generator
            Итератор по параметрам всех слоев.
        """
        for module in self.modules:
            params = module.parameters()
            if isinstance(params, tuple):
                yield from params
            else:
                yield params

    def zero_grad(self):
        """Обнуляет все накопленные градиенты во всех слоях."""
        for module in self.modules:
            module.zero_grad()

    def _compute_gradients(self, grad):
        """
        Вычисляет градиенты для всех слоев модели в обратном порядке.

        Параметры:
        ----------
        grad: np.ndarray
            Градиент функции ошибки по выходу модели.

        Возвращает:
        -----------
        np.ndarray
            Градиент по входу первого слоя.
        """
        for module in reversed(self.modules):
            grad = module._compute_gradients(grad)
        return grad

    def train(self):
        """Переводит модель в режим обучения (training mode)."""
        for module in self.modules:
            module.train()

    def eval(self):
        """Переводит модель в режим инференса (evaluation mode)."""
        for module in self.modules:
            module.eval()

    def __repr__(self):
        """
        Возвращает строковое представление модели.
        """
        module_str = ",\n    ".join(map(str, self.modules))
        return f"Sequential(\n    {module_str}\n)"