import numpy as np
import time
import quandl
import pandas as pd

class Geometric_BM():

    def __init__(self,number_inputs,time_window):


        # Parameter Assignments
        self.So = 10
        self.dt = 1  # day   # User input
        self.T = time_window
        self.N = self.T / self.dt
        self.mu = 0
        self.sigma = 0.1
        self.t = np.arange(1, int(self.N) + 1)

        self.scen_size = number_inputs  # User input

        #Brownian path
        self.b, self.W = self.Brownian()
        self.drift, self.diffu = self.drift_diffusion()
        self.S = self.predict_path()


    def Brownian(self):

        dt = 0.01

        b = {str(scen): np.random.normal(0, 1, int(self.N)) for scen in range(1, self.scen_size + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, self.scen_size + 1)}

        return b, W

    def drift_diffusion(self):

        drift = (self.mu - 0.5 * self.sigma ** 2) * self.t
        diffusion = {str(scen): self.sigma * self.W[str(scen)] for scen in range(1, self.scen_size + 1)}

        return drift, diffusion

    def predict_path(self):

        S = np.array([self.So * np.exp(self.drift + self.diffu[str(scen)]) for scen in range(1, self.scen_size + 1)])
        S = np.hstack((np.array([[self.So] for scen in range(self.scen_size)]), S))  # add So to the beginning series

        return S

class Orn_Uh():

    def __init__(self,number_inputs,unknown_days):
        # Parameter Assignments
        self.So = 10
        self.dt = 1  # day   # User input
        self.T = unknown_days
        self.N = self.T / self.dt
        self.mu = 0 #mean
        self.sigma = 0.1 # Standard deviation.
        self.t = np.arange(1, int(self.N) + 1)
        self.tau = 0.05 #time constant


