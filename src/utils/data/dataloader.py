import numpy as np

class DataLoader:
    """
    Загрузчик данных для итеративной подачи батчей в модель.

    ---------
    Параметры
    ---------
    dataset : list
        Датасет, состоящий из пар (вектор, метка).

    batch_size : int, optional, default=1000
        Размер батча (количество элементов в одном батче).

    is_train : bool, optional, default=True
        Если True, перед выдачей батчей перемешивает датасет.
        Если False, данные не перемешиваются.
    """

    def __init__(self, dataset, batch_size=128, shuffle=False, drop_last=False):
        self.dataset = dataset  # Датасет
        self.batch_size = batch_size  # Размер батча
        self.shuffle = shuffle  # Режим обучения (перемешивание данных)
        self.drop_last = drop_last

        self.init_array()

    def init_array(self):
        self.array = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.array)
        return self.array

    def __iter__(self):
        """
        Возвращает итератор для обхода датасета батчами.
        """
        return self

    def __len__(self):
        """
        Возвращает количество батчей в датасете.
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __next__(self):
        """
        Возвращает следующий батч данных.

        Возвращает:
        ----------
        data : np.array
            Массив данных (векторы) из текущего батча.
        labels : list
            Список меток, соответствующих данным из текущего батча.

        Исключения:
        ----------
        StopIteration
            Если все данные уже были возвращены.
        """
        if len(self.array) == 0:
            self.init_array()
            raise StopIteration()  # Если данные закончились, завершаем итерацию

        if len(self.array) < self.batch_size and self.drop_last:
            self.init_array()
            raise StopIteration()

        # Выбираем индексы для текущего батча
        if len(self.array) > self.batch_size:
            selected = self.array[:self.batch_size]  # Берём первые batch_size элементов
            self.array = self.array[self.batch_size:]  # Убираем выбранные элементы
        else:
            selected = self.array  # Берём оставшиеся элементы
            self.array = []  # Очищаем массив индексов

        # Собираем данные и метки для текущего батча
        data = [self.dataset[ind][0] for ind in selected]
        labels = [self.dataset[ind][1] for ind in selected]

        return np.array(data, dtype=np.float32), np.array(labels)  # Возвращаем батч