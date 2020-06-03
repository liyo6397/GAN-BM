import numpy as np
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM
import scipy

class joint_distribution():

    def __init__(self, nx, ny, mu, sigma, s0, paths):


        self.s0 = s0
        self.Nx = nx
        self.Ny = ny
        self.paths = paths
        self.num_paths = paths.shape[0]
        self.T = paths.shape[1]

        self.mu = mu
        self.sigma = sigma

    def BM_joint_dstr(self,s, s0, path, delta_t):

        var = self.sigma**2
        dom = (np.sqrt(2*np.pi))**len(s)*np.sqrt((delta_t)**len(s))
        factor_sum = 0



        for i in range(len(s)-1):
            factor_sum += ((((1/self.sigma)*np.log(s[i+1]/s[i])-0.5*delta_t*var))**2)/(delta_t)



        factor_sum = -factor_sum*0.5

        factor = np.exp(factor_sum)

        pdf = factor/dom

        print(factor)
        print(dom)

        return pdf










