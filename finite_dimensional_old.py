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

    def setup_coord_value(self):

        y_max = max(self.paths[:, -1])
        y_min = min(self.paths[:, -1])

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


    def y_grid_values(self, time_steps):



        s_t = np.zeros((time_steps+1, self.num_paths))
        time_st = np.zeros(time_steps+1)
        grid_x_diff = time_steps / (self.T - 1)
        h_st = 1/grid_x_diff

        int_grid_x_diff = int(grid_x_diff)

        for j in range(self.num_paths):
            for i in range(self.T-1):

                T_idx = i*int_grid_x_diff

                s_t[T_idx, j] = self.paths[j, i]
                tan = self.paths[j, i + 1] - self.paths[j, i]
                time_st[T_idx] = h_st * (T_idx)

                for k in range(1, int_grid_x_diff):
                    time_st[T_idx+k] = h_st*(T_idx+k)
                    if tan >= 0:
                        s_t[T_idx + k, j] = tan * k / int_grid_x_diff + self.paths[j, i]  # y=ax+b
                    else:
                        s_t[T_idx + k, j] = -tan * (int_grid_x_diff-k) / int_grid_x_diff + self.paths[j, i+1]  # y=ax+b


        s_t[- 1, :] = self.paths[:, -1]
        time_st[-1] = h_st * (T_idx+grid_x_diff)


        return s_t, time_st

    def fd_theoritical_1(self, s, delta_t, s0):

        s0 = s[0]
        exp_term1 = 0
        dom1 = 1

        for d in range(self.num_paths):
            mean = np.log(s0) + self.mu * 1
            var = self.sigma * np.sqrt(1)
            std = np.sqrt(var)

            exp_term1 += (np.log(s0)-mean)**2/(2*std**2)
            dom1 = dom1*s[0]*std*np.sqrt(2*np.pi)

        term1 = np.exp(-exp_term1)/dom1


        exp_term2 = 0
        dom2 = 1
        delta_t = 1

        Nt = len(s)

        for t in range(Nt-1):
            mean = np.log(s0) + self.mu * t*delta_t
            var = self.sigma * np.sqrt(t*delta_t)
            std = np.sqrt(var)

            H = self.mu*delta_t

            exp_term2 += ((np.log(s[t+1]/s[t]) - H) ** 2) / (2 * self.sigma ** 2*delta_t)
            dom2 = dom2 * s[t] * self.sigma * np.sqrt(2 * np.pi * delta_t)


        term2 = np.exp(-exp_term2)/ dom2






        pdf = term1*term2


        return pdf

    def num_pts_grid(self, grid_st, time_st):



        dim_grid_st = grid_st.shape[1]

        grid = np.zeros((self.Nx,self.Ny))



        count = 0


        for i in range(self.Nx-1):
            for j in range(self.Ny-1):
                for t in range(time_st.shape[0]):
                    if time_st[t] >= self.time[i] and time_st[t] <= self.time[i+1]:
                        for dim in range(dim_grid_st):
                            if grid_st[t,dim] >= self.y[j]  and grid_st[t,dim] <= self.y[j+1]:
                                grid[i,j] += 1
                                count += 1



        return grid

    def fd_theoritical_2(self, s, delta_t):

        s0 = s[0]
        exp_term1 = 0
        dom1 = 1

        #for d in range(self.num_paths):
        mean = 0
        std=1

        exp_term1 += (np.log(s0)-mean)**2/(2*std**2)
        dom1 = dom1*s[0]*std*np.sqrt(2*np.pi)

        term1 = np.exp(-exp_term1)/dom1


        exp_term2 = 0
        dom2 = 1


        Nt = len(s)

        for t in range(Nt-1):
            #mean = np.log(s0) + self.mu * t*delta_t
            #var = self.sigma * np.sqrt(t*delta_t)
            #std = np.sqrt(var)

            H = self.mu*delta_t

            exp_term2 += ((np.log(s[t+1]/s[t])) ** 2) / (2 *delta_t)
            dom2 = dom2 * (s[t+1]/s[t]) * np.sqrt(2 * np.pi * delta_t)


        term2 = np.exp(-exp_term2)/ dom2






        pdf = term1*term2

        return pdf

    def grid_density(self, grids_density, start, end ,num_step):

        sum_row = np.sum(grids_density)
        sum_all = np.sum(sum_row)

        sum_den = np.sum(grids_density[:, start:end])
        pdf = np.sum(sum_den) / sum_all



        return pdf

    def fd_theoritical_3(self,s, delta_t, s0):


        term1 = 1/np.sqrt((2*np.pi)**(self.paths.shape[1])*delta_t**(self.paths.shape[1]-1))

        exp_term = 0
        for i in range(0, len(s)):
            exp_term += 0.5*(s[i]-s[i-1])**2/delta_t

        pdf = term1*np.exp(-exp_term)

        return pdf

    def cov_multi(self,s, s0):

        N = len(s)
        cov = np.zeros((N,N))

        for i in range(N):
            mean_row = np.log(s0) + self.mu * (i+1)
            for j in range(N):
                mean_col = np.log(s0) + self.mu * (j+1)
                cov[i,j] = (np.log(s[i]) - mean_row)*(np.log(s[j]) - mean_col)

        return cov

    def fd_multivar(self,s):

        cov = self.cov_multi(s,self.paths[0,0])
        #cov = 1
        term1_1 = np.sqrt((2*np.pi)**(len(s)))
        term1_2 = np.sqrt(cov)

        term1 = 1/(term1_1*term1_2)

        term2 = 1/np.sqrt(np.linalg.det(cov))
        print(np.linalg.det(cov))

        s = s-self.mu
        s_t = s.T
        inv_cov = 1/cov
        e_term = s_t.dot(inv_cov)
        e_term = e_term.dot(s)
        exp_term = np.exp(-0.5*e_term)

        term3 = 1
        for j in range(len(s)):
            term3 = term3*1/s[j]*exp_term

        pdf = term1*term2*term3

        return pdf

    def central_limit(self,s1,s0,time_range):

        z_score = ((1/self.sigma)*(np.log(s1/s0)-self.mu-0.5*self.sigma**2*10))/(np.sqrt(time_range))
        p_value = scipy.stats.norm.cdf(z_score)

        return p_value

    def BM_joint_dstr(self,s, s0, path, delta_t):

        var = self.sigma**2
        dom = (np.sqrt(2*np.pi))**len(s)*np.sqrt((delta_t)**len(s))
        #factor_sum = (np.log(s[0])-0.5*var)**2/(var*delta_t)
        #print(factor_sum)
        factor_sum = 0



        for i in range(len(s)-1):
            factor_sum += ((((1/self.sigma)*np.log(s[i+1]/s[i])-0.5*delta_t*var))**2)/(delta_t)



        factor_sum = -factor_sum*0.5

        factor = np.exp(factor_sum)

        pdf = factor/dom

        print(factor)
        print(dom)

        return pdf

    def grid_prob(self, data, time_st, target1, target2, time1, time2, start):

        count = 0
        total = 0

        #for i in range(2):
        for t in range(time_st.shape[0]):
            for j in range(data.shape[1]):
                if time_st[t] >= time1 and time_st[t] <= time2:
                    total += 1
                    #for dim in range(dim_grid_st):
                    if data[t,j] >= target1  and data[t, j] <= target2:
                        count += 1
                #if data[start+i,j] >= target1 and data[start+i,j] <= target2:
                #    count += 1

        #pdf_data = (count / ((data.shape[0]*data.shape[1])))#*((np.sum(data.shape[1]))/(data.shape[0]*data.shape[1]))
        pdf_data = count/total



        return pdf_data


    def grid_loss(self):


        time = np.linspace(0, self.paths.shape[1]-1, self.Nx)
        y = np.linspace(min(self.paths[:,-1]), max(self.paths[:,-1]), self.Ny)
        p_values = np.zeros((self.Nx-1, self.Ny-1))

        #X, Y = np.meshgrid(time, y)
        #Z = self.central_limit(X,self.s0,Y)
        X = np.linspace(0,time[-1],self.Nx-1)
        Y = np.linspace(0,time[-1],self.Ny-1)
        X, Y = np.meshgrid(X, Y)
        p_grid = np.zeros_like(p_values)

        num_steps = 100
        data, time_st = self.y_grid_values(num_steps)


        for i in range(1,self.Nx-1):
            #X[i] = time[i]
            for j in range(self.Ny-1):
                p1_value = self.central_limit(y[j], self.s0, time[i])
                p2_value = self.central_limit(y[j+1], self.s0, time[i+1])

                p = p2_value - p1_value
                p_values[i,j] = p

                grid_prob = self.grid_prob(data, time_st, y[j], y[j + 1], time[i], time[i+1], i)
                p_grid[i,j] = grid_prob

                #Y = y[j]

        return X, Y, p_values, p_grid




































































