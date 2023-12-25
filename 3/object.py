# -*- coding: utf-8 -*
import numpy as np
import numpy.typing as npt
from typing import List
import matplotlib.pyplot as plt
from tools import *
from sources import Probe, Source, AnimateFieldDisplay, showProbeSignals

'''* импорт с получением доступа ко всем именам'''


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':
    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Параметры моделирования
    # Частота сигнала, Гц
    f_Hz = 1.5e9

    # Дискрет по пространству в м
    dx = 4e-3

    wavelength = c / f_Hz
    Nl = wavelength / dx

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 3.5

    # Время расчета в секундах
    maxTime_s = 20e-9

    # Положение источника в м
    sourcePos_m = 2

    # Координаты датчиков для регистрации поля в м
    probesPos_m = [2]

    # Параметры слоев
    layers_cont = [LayerContinuous(0, eps=8.0, sigma=0.0)]

    # Скорость обновления графика поля
    speed_refresh = 15

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    # Вывод параметров моделирования
    print(f'Число Куранта: {Sc}')
    print(f'Размер области моделирования: {maxSize_m} м')
    print(f'Время расчета: {maxTime_s * 1e9} нс')
    print(f'Координата источника: {sourcePos_m} м')
    print(f'Частота сигнала: {f_Hz * 1e-9} ГГц')
    print(f'Длина волны: {wavelength} м')
    print(f'Количество отсчетов на длину волны (Nl): {Nl}')
    probes_m_str = ', '.join(['{:.6f}'.format(pos) for pos in probesPos_m])
    print(f'Дискрет по пространству: {dx} м')
    print(f'Дискрет по времени: {dt * 1e9} нс')
    print(f'Координаты пробника [м]: {probes_m_str}')
    print()
    print(f'Размер области моделирования: {maxSize} отсч.')
    print(f'Время расчета: {maxTime} отсч.')
    print(f'Координата источника: {sourcePos} отсч.')
    probes_str = ', '.join(['{}'.format(pos) for pos in probesPos])
    print(f'Координаты пробника [отсч.]: {probes_str}')

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Коэффициенты для учета потерь
    loss = sigma * dt / (2 * eps * eps0)
    ceze = (1.0 - loss) / (1.0 + loss)
    cezh = W0 / (eps * (1.0 + loss))

    # Расчет коэффициентов для граничных условий
    # Sc' для левой границы
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Источник
    magnitude = 1.0
    signal = GaussianDiff(1, 150, 50)

    source = Source(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = AnimateFieldDisplay(maxSize, display_ymin, display_ymax, display_ylabel, dx, dt)

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Граничное условие PMC (справа)
        Hy[-1] = 0

        # Источник возбуждения
        Hy[sourcePos - 1] += source.getFieldH(t)
        # Ez[1] в предыдущий момент времени
        oldEzLeft = Ez[1]
        # Ez[-2] в предыдущий момент времени
        # oldEzRight = Ez[-2]
        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1: -1] * Ez[1: -1] + cezh[1: -1] * (Hy[1:] - Hy[: -1])
        # Граничные условия ABC первой степени (слева)
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]

        # Источник возбуждения
        Ez[sourcePos] += source.getFieldE(t)

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)
            display.stop()

    # Отображение сигнала, сохраненного в пробнике
    showProbeSignals(probes, dx, dt, -2.1, 2.1)

    # Построение спектра
    plt.figure()
    sp = np.fft.fft(probes[0].E)
    freq = np.fft.fftfreq(maxTime)
    plt.plot(freq / (dt * 1e9), abs(sp) / max(abs(sp)))
    plt.xlim(0.1, 2.2)
    plt.grid()
    plt.xlabel('f, ГГц')
    display.stop()
    plt.show()