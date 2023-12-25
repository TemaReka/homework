# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Probe:
    '''
    Класс для хранения временного сигнала в датчике.
    '''

    def __init__(self, position: int, maxTime: int):
        '''
        position - положение датчика (номер ячейки).
        maxTime - максимальное количество временных
            шагов для хранения в датчике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: npt.NDArray[float], H: npt.NDArray[float]):
        '''
        Добавить данные по полям E и H в датчик.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str,
                 dx: float,
                 dt: float,
                 title: Optional[str] = None):
        '''
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        '''
        self.maxXSize = maxXSize
        print(self.maxXSize)
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'
        self._dx = dx
        self._dt = dt
        self._title = title

        '''
        dx - дискрет по пространству, м
        dt - дискрет по времени, с (они используются позже в мэйн файле, где задаются все данные, а также ниже)
        '''

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = np.arange(self.maxXSize) * self._dx

        # Включить интерактивный режим для анимации
        plt.ion()

        # Создание окна для графика
        self._fig, self._ax = plt.subplots(figsize=(10, 6.5))

        if self._title is not None:
            self._fig.suptitle(self._title)

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self._dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать датчики.

        probesPos - список координат датчиков для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение датчиков
        self._ax.plot(np.array(probesPos) * self._dx,
                      [0] * len(probesPos), self._probeStyle)

        for n, pos in enumerate(probesPos):
            self._ax.text(
                pos * self._dx,
                0,
                '\n{n}'.format(n=n + 1),
                verticalalignment='top',
                horizontalalignment='center')

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение источников
        self._ax.plot(np.array(sourcesPos) * self._dx, [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self._dx, position * self._dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        plt.ioff()

    def updateData(self, data: npt.NDArray[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount * self._dt * 1e9))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def showProbeSignals(probes: List[Probe], dx: float, dt: float, minYSize: float, maxYSize: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

    # Создание окна с графиков
    fig, ax = plt.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E) * dt * 1e9)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('t, нс')
    ax.set_ylabel('Ez, В/м')
    ax.grid()

    time_list = np.arange(len(probes[0].E)) * dt * 1e9

    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(time_list, probe.E)

    # Показать окно с графиками
    plt.show()


class Source1D(metaclass=ABCMeta):
    '''
    Базовый класс для всех источников одномерного метода FDTD
    '''

    @abstractmethod
    def getFieldE(self, position, time):
        '''
        Метод должен возвращать значение поля источника в момент времени time
        '''
        pass

    def getFieldH(self, time):
        return 0.0


class SourcePlaneWave(metaclass=ABCMeta):
    @abstractmethod
    def getFieldE(self, position, time):
        pass


class Source(Source1D):
    def __init__(self, source: SourcePlaneWave,
                 sourcePos: float,
                 Sc: float = 1.0,
                 eps: float = 1.0,
                 mu: float = 1.0):
        self.source = source
        self.sourcePos = sourcePos
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.W0 = 120.0 * np.pi

    def getFieldH(self, time):
        return -(self.Sc / (self.W0 * self.mu)) * (
                self.source.getFieldE(self.sourcePos, time) - self.source.getFieldE(self.sourcePos - 1, time))

    def getFieldE(self, time):
        return (self.Sc / np.sqrt(self.eps * self.mu)) * (
                self.source.getFieldE(self.sourcePos - 0.5, time + 0.5) + self.source.getFieldE(self.sourcePos + 0.5,
                                                                                                time + 0.5))