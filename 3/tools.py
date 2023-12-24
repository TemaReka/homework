import math
import numpy as np
from Animation import Source1D



class Gaussian(Source1D):
    '''
    Источник, создающий гауссов импульс
    '''

    def __init__(self, magnitude, dg, wg):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.magnitude = magnitude
        self.dg = dg
        self.wg = wg

    def getE(self, v, time):
        return self.magnitude * np.exp(-((time - self.dg) / self.wg) ** 2)

class LayerContinuous:
    def __init__(self,
                 xmin: float,
                 xmax: float = None,
                 eps: float = 8.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma

class LayerDiscrete:
    def __init__(self,
                 xmin: int,
                 xmax: int = None,
                 eps: float = 8.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)

def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)

