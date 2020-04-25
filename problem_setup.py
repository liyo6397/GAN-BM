import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM, Orn_Uh
import scipy.stats as stats


class problem_setup():

    def __init__(self):
        self.number_inputs = 100
        self.converge_crit = 10 ** (-6)
        self.print_itr = 1000


    def Brownian_motion(self):

        self.unknown_days = 10
        self.mu = 0
        self.sigma = 0.1
        self.data_type = "Geometric Brownian Motion"
        self.method = "uniform"

        gbm = Geometric_BM(self.number_inputs, self.unknown_days, self.mu, self.sigma)
        paths = gbm.predict_path()

        return paths

