import numpy as np
import time
#import quandl
import pandas as pd
import matplotlib.pyplot as plt

class Geometric_BM():

    def __init__(self,number_inputs,time_window,mu,sigma, s0):


        # Parameter Assignments
        self.s0 = s0
        self.dt = 1  # day   # User input
        self.T = time_window
        self.N_days = self.T / self.dt
        self.mu = mu
        self.sigma = sigma
        self.t = np.arange(1, int(self.N_days) + 1)

        self.scen_size = number_inputs  # User input

        #Brownian path
        self.b, self.W = self.Brownian()
        self.drift, self.diffu = self.drift_diffusion()
        self.S = self.predict_path()


    def Brownian(self):

        dt = 0.01

        b = {str(scen): np.random.normal(0, 1, int(self.N_days)) for scen in range(1, self.scen_size + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, self.scen_size + 1)}

        return b, W

    def drift_diffusion(self):

        drift = (self.mu - 0.5 * self.sigma ** 2) * self.t
        diffusion = {str(scen): self.sigma * self.W[str(scen)] for scen in range(1, self.scen_size + 1)}

        return drift, diffusion

    def predict_path(self):

        S = np.array([self.s0 * np.exp(self.drift + self.diffu[str(scen)]) for scen in range(1, self.scen_size + 1)])
        S = np.hstack((np.array([[self.s0] for scen in range(self.scen_size)]), S))  # add So to the beginning series
        #S = np.array([np.exp(self.drift + self.diffu[str(scen)]) for scen in range(1, self.scen_size + 1)])
        #S = np.hstack((np.array([[self.So] for scen in range(self.scen_size)]), S))  # add So to the beginning series

        return S

class Orn_Uh():

    def __init__(self,number_inputs,unknown_days, mu, sigma, theta, s0, t0, tend):
        # Parameter Assignments
        self.s0 = s0
        self.dt = 1  # day   # User input
        self.num_ipts = number_inputs
        self.T = unknown_days
        self.N_days = self.T / self.dt
        self.mu = mu #mean
        self.sigma = sigma # Standard deviation.
        self.t = np.linspace(0,self.T,self.T)
        self.theta = theta #time constant
        self.dt = np.mean(np.diff(self.t))
        self.ts = np.linspace(t0,tend,unknown_days)
        #self.ts = np.arange(1, int(self.N_days) + 1)
        self.et = np.exp(-self.theta * self.ts)

        # Brownian path
        self.drift, self.diffusion = self.drift_diffusion()

    def Brownian(self):

        #radn = np.random.normal(0, 1, int(self.N_days))
        b = {str(scen): np.exp(self.theta*self.ts)*np.random.normal(0, 1, int(self.N_days))*np.sqrt(self.dt) for scen in range(1, self.num_ipts + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, self.num_ipts + 1)}

        return W


    def drift_diffusion(self):


        W = self.Brownian()

        drift = self.mu*(1-self.et)
        diffusion = {str(scen): self.sigma * self.et * W[str(scen)] for scen in range(1, self.num_ipts + 1)}

        return drift, diffusion



    def predict_path(self):


        S = np.array([self.s0*self.et + self.drift+self.sigma*self.diffusion[str(scen)] for scen in range(1, self.num_ipts + 1)])
        S = np.hstack((np.array([[self.s0] for scen in range(self.num_ipts)]), S))  # add So to the beginning series


        return S


if __name__ == "__main__":
    number_inputs = 10
    unknown_days = 30
    mu = 0.8
    sigma = 0.3
    theta = 1.1
    s0 = 1
    t0 = 0
    tend = 2
    #gbm = Geometric_BM(number_inputs, unknown_days,mu,sigma,s0)
    #b,W = gbm.Brownian()
    #S = np.array([W[str(scen)] for scen in range(1, number_inputs + 1)])


    orn = Orn_Uh(number_inputs, unknown_days,mu,sigma, theta, s0, t0, tend)
    paths = orn.predict_path()
    #print(paths)

    #print(orn.et)

    for i in range(number_inputs):
        plt.title("Ornstein-Uhlenbeck process")
        plt.plot(paths[i, :])
        print(orn.drift)
        plt.ylabel('Sample values')
        plt.xlabel('Time')

    plt.show()



