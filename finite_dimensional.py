import numpy as np
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM

class joint_distribution():

    def __init__(self, paths, nx, ny):

        self.paths = paths
        self.num_paths = paths.shape[0]
        self.T = paths.shape[1]
        self.Nx = nx
        self.Ny = ny


        self.time, self.y = self.setup_coord_value()

        self.hx = (self.time[-1]-self.time[0])/(self.Nx)
        self.hy = (self.y[-1] - self.y[0]) / (self.Ny)

    def setup_coord_value(self):

        y_max = max(self.paths[:, -1]) + 0.01
        y_min = min(self.paths[:, -1]) - 0.01

        t = np.linspace(0, self.T, self.Nx)
        y = np.linspace(y_min, y_max, self.Ny)

        return t,y

    def grid_boundary(self, t_idx, y_idx):

        t_left = self.time[t_idx]
        t_right = self.time[t_idx+1]

        y_top = self.y[y_idx]
        y_bottom = self.y[y_idx + 1]

        return t_left, t_right, y_top, y_bottom

    def tangent_timezone(self):

        tan = np.zeros((self.T-1, self.num_paths-1))

        for i in range(self.T-1):
            for j in range(self.num_paths-1):
                tan[i,j] = (self.paths[j+1,i+1]-self.paths[j,i])

        return tan

    def y_grid_values(self):

        tan = self.tangent_timezone()
        s_t = np.zeros((self.Nx, self.num_paths))

        for i, day in enumerate(self.time):
            #t_range = self.time[i + 1] - self.time[i]
            t = int(day)
            path = self.paths[t,:]
            print(day)
            for j in range(self.Ny-1):
                s = int(self.y[j]-self.paths[t,0])
                start = self.y[s]
                if t <= tan.shape[0]-1:
                    sn = start+day*tan[t, s]
                else:
                    sn = start+day*tan[-1, s]
                s_t[i,j] = sn

        return s_t

    def y_grid_values_new(self):

        grid_x_diff = int(self.Nx/(self.T))
        s_t = np.zeros((self.Nx, self.Ny))

        for j in range(self.num_paths):
            for i in range(self.T-1):
                s_t[i,j] = self.paths[j,i]
                T_idx = i*grid_x_diff

                for k in range(1,grid_x_diff):
                    day = int(self.time[T_idx+k])
                    if day <= self.paths.shape[1]-1:
                        tan = self.paths[j, i + 1] - self.paths[j, i]
                    s_t[T_idx+k,j] = tan*self.time[T_idx+k] + self.paths[j,i] # y=ax+b


        return s_t




























