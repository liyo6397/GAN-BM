import numpy as np
from plot_result import plot_result
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM
import scipy
from scipy import integrate

class joint_distribution():

    def __init__(self, nx, ny, mu, sigma, s0):


        self.s0 = s0
        self.Nx = nx
        self.Ny = ny
        #self.paths = paths
        #self.num_paths = paths.shape[0]
        #self.T = paths.shape[1]


        self.mu = mu
        self.sigma = sigma



    def cubid_probability(self):

        time = np.linspace(0, self.paths.shape[1], self.Nx)
        values = np.linspace(0, self.paths.shape[0], self.Ny)
        col_count = np.zeros((self.Nx, self.Ny))

        for i in range(self.Nx-1):
            for j in range(self.Ny-1):
                count = self.cubic_count(time[i],time[i+1],values[j],values[j+1])
                col_count[i,j] = count

        cubid_prob = np.zeros_like(col_count)
        sum_col_count = np.sum(col_count)
        for i in range(self.Nx-1):
            for j in range(self.Ny-1):
                cubid_prob[i,j] = col_count[i,j]/sum_col_count

        return cubid_prob

    def tan(self,x0, x1, y0, y1):

        tan = (x1-x0)/(y1-y0)

        return tan

    def cubic_dens(self,x0,x1,time_steps):


        day_prob = np.zeros(self.paths.shape[1]-1)
        #tan = self.tan(x0,x1,y0,y1)
        delta_t = self.T/self.paths.shape[1]
        #ver_top = int(y1/delta_t)+1
        #ver_bottom = int(y0 / delta_t)

        s_t, time_st = self.y_grid_values(time_steps)
        for j in range(1,self.paths.shape[1]-1):
            count = 0
            for i in range(self.paths.shape[0]):
                if self.paths[i,j] >= x0 and self.paths[i,j] <= x1:
                    count += 1

            day_prob[j] = count/self.paths.shape[0]

        #pdf = count / (self.paths.shape[0]*self.paths.shape[1])
        pdf = day_prob[1]
        for i in range(1,len(day_prob)):
            pdf = day_prob[i]*pdf

        return pdf

    def y_grid_values(self, time_steps, paths):

        T = paths.shape[1]
        num_paths = paths.shape[0]

        s_t = np.zeros((num_paths, time_steps+1))
        time_st = np.zeros(time_steps+1)
        grid_x_diff = time_steps / (T - 1)
        h_st = 1/grid_x_diff

        int_grid_x_diff = int(grid_x_diff)

        for i in range(num_paths):
            for j in range(T-1):

                T_idx = j*int_grid_x_diff

                s_t[i, T_idx] = paths[i, j]
                tan = paths[i, j + 1] - paths[i, j]
                time_st[T_idx] = h_st * (T_idx)

                for k in range(1, int_grid_x_diff):
                    time_st[T_idx+k] = h_st*(T_idx+k)
                    if tan >= 0:
                        s_t[i, T_idx + k] = tan * k / int_grid_x_diff + paths[i, j]  # y=ax+b
                    else:
                        s_t[i, T_idx + k] = -tan * (int_grid_x_diff-k) / int_grid_x_diff + paths[i, j+1]  # y=ax+b


        s_t[:, -1] = paths[:, -1]
        time_st[-1] = h_st * (T_idx+grid_x_diff)


        return s_t, time_st

    def BM_joint_dstr2(self):

        numBins = 2  # number of bins in each dimension
        jointProbs, edges = np.histogramdd(self.paths, bins=numBins)
        #jointProbs /= jointProbs.sum()

        return edges

    def grid_prob(self, data, time_st, target1, target2, time1, time2):

        count = 0
        total = 0

        # for i in range(2):
        for j in range(data.shape[1]):
            if time_st[j] >= time1 and time_st[j] <= time2:
                #total += 1
                for i in range(data.shape[0]):
                    if data[i,j] >= target1 and data[i, j] <= target2:
                        count += 1

        pdf_data = count / (data.shape[0]*data.shape[1])
        #pdf_data = count / np.sum(data)

        return pdf_data

    def grid_loss(self,data_1, data_2, time_st):

        loss_grid = np.zeros((self.Nx-1, self.Ny-1))
        min_data1 = min(data_1[:,-1])
        min_data2 = min(data_2[:, -1])
        max_data1 = max(data_1[:, -1])
        max_data2 = max(data_2[:, -1])
        y = np.linspace(min(min_data1, min_data2), max(max_data1, max_data2), self.Ny)

        #prob_theo = np.zeros((self.Nx-1)*(self.Ny-1))
        #prob_emp = np.zeros((self.Nx - 1) * (self.Ny - 1))

        for i in range(self.Nx-1):
            target1 = y[i]
            target2 = y[i + 1]
            for j in range(self.Ny-1):
                time1 = j
                time2 = j+1

                prob_theo = self.grid_prob(data_1, time_st, target1, target2, time1, time2)
                prob_emp = self.grid_prob(data_2, time_st, target1, target2, time1, time2)


                loss = np.abs(prob_theo - prob_emp)
                loss_grid[i,j] = loss




        return loss_grid

    def meshgrid_prob(self, X, Y, time1, time2, target1, target2):

        count = 0
        total = 0

        # for i in range(2):
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if X[i] >= time1 and X[i] <= time2:
                    total += 1
                    if Y[j] >= target1 and data[i, j] <= target2:
                        count += 1

        pdf_data = count / (data.shape[0] * data.shape[1])

        return pdf_data

    def meshgrid_loss(self, data_1, data_2, time_st):

        x = np.linspace(time_st[0], time_st[-1], self.Nx)
        min_data1 = min(map(min, data_1))
        min_data2 = min(map(min, data_2))
        max_data1 = max(map(max, data_1))
        max_data2 = max(map(max, data_2))
        y1 = np.linspace(min_data1, max_data1, self.Ny)
        #y2 = np.linspace(min_data2, max_data2, self.Ny)
        #X, Y1 = np.meshgrid(x, y1)
        #X, Y2 = np.meshgrid(x, y2)

        prob_theo = np.zeros((self.Nx - 1, self.Ny - 1))
        prob_emp = np.zeros((self.Nx - 1, self.Ny - 1))
        loss_grids = np.zeros_like(prob_theo)


        for j in range(self.Nx-1):
            time1 = x[j]
            time2 = x[j + 1]
            for i in range(self.Ny-1):
                target1 = y1[i]
                target2 = y1[i + 1]

                prob_theo[i,j] = self.grid_prob(data_1, time_st, target1, target2, time1, time2)
                prob_emp[i,j] = self.grid_prob(data_2, time_st, target1, target2, time1, time2)

                loss = np.abs(prob_theo[i,j] - prob_emp[i,j])
                loss_grids[i,j] = loss


        return prob_theo, prob_emp, loss_grids

    def grid_loss_scatter(self,data_1, data_2, time_st):

        loss_grid = np.zeros((self.Nx-1, self.Ny-1))
        min_data1 = min(data_1[:,-1])
        min_data2 = min(data_2[:, -1])
        max_data1 = max(data_1[:, -1])
        max_data2 = max(data_2[:, -1])
        y = np.linspace(min(min_data1, min_data2), max(max_data1, max_data2), self.Ny)

        prob_theo = np.zeros((self.Nx-1, self.Ny-1))
        prob_emp = np.zeros((self.Nx - 1, self.Ny - 1))

        for i in range(self.Nx-1):
            target1 = y[i]
            target2 = y[i + 1]
            for j in range(self.Ny-1):
                time1 = j
                time2 = j+1

                prob_theo[i,j] = self.grid_prob(data_1, time_st, target1, target2, time1, time2)
                prob_emp[i, j] = self.grid_prob(data_2, time_st, target1, target2, time1, time2)


                loss = np.abs(prob_theo[i,j] - prob_emp[i,j])
                loss_grid[i,j] = loss




        return prob_theo, prob_emp, loss_grid

    def fdd_den_by_value(self, prob_theo, prob_emp):

        theo = np.zeros(prob_theo.shape[0])
        emp = np.zeros(prob_emp.shape[0])

        for i in range(prob_theo.shape[0]):
            theo[i] = np.sum(prob_theo[i,:])
            emp[i] = np.sum(prob_emp[i, :])

        return theo, emp

    def MAPE(self, den_1, den_2):

        loss_sum = 0
        N = den_1.shape[0]
        M = den_1.shape[1]

        count = 0

        for i in range(N):
            for j in range(M):
                if den_1[i,j] != 0:
                    loss_sum += np.abs(den_1[i,j]-den_2[i,j])/den_1[i,j]
                #elif den_1[i,j] == 0 and den_2[i,j] != 0:
                #    loss_sum += np.abs(den_1[i,j]-den_2[i,j])/den_2[i,j]
                #else:
                #    count += 1

        print(loss_sum)

        mape = loss_sum/(N*M)

        return mape

    def Relative_Percent_Difference(self, den_1, den_2):

        loss_sum = 0
        N = den_1.shape[0]
        M = den_1.shape[1]


        for i in range(N):
            for j in range(M):
                if den_1[i,j] == 0 or den_2[i,j] == 0:
                    continue
                else:
                    #loss_sum += 2*(den_1[i,j] - den_2[i,j])/(np.abs(den_1[i,j]) + np.abs(den_2[i,j]))
                    loss_sum += 2 * (den_1[i, j] - den_2[i, j]) / (np.abs(den_1[i, j]) + np.abs(den_2[i, j]))


        diff = loss_sum/(N*M)

        return diff

    def TheoEmp_loss(self, data_1, data_2, time_st):

        x = np.linspace(time_st[0], time_st[-1], self.Nx)
        min_data1 = min(map(min, data_1))
        min_data2 = min(map(min, data_2))
        max_data1 = max(map(max, data_1))
        max_data2 = max(map(max, data_2))
        y1 = np.linspace(min_data1, max_data1, self.Ny)
        #y2 = np.linspace(min_data2, max_data2, self.Ny)
        #X, Y1 = np.meshgrid(x, y1)
        #X, Y2 = np.meshgrid(x, y2)

        prob_theo = np.zeros((self.Nx - 1, self.Ny - 1))
        prob_emp = np.zeros((self.Nx - 1, self.Ny - 1))
        loss_grids = np.zeros_like(prob_theo)

        delta_t = 10/self.Nx


        for j in range(self.Nx-1):
            time1 = x[j]
            time2 = x[j + 1]
            for i in range(self.Ny-1):
                target1 = y1[i]
                target2 = y1[i + 1]

                prob_theo[i,j] = self.BM_joint_dstr(data_2, time_st, delta_t, target1, target2, time1, time2)
                prob_emp[i,j] = self.grid_prob(data_2, time_st, target1, target2, time1, time2)

                loss = np.abs(prob_theo[i,j] - prob_emp[i,j])
                loss_grids[i,j] = loss

        #print(np.sum(prob_theo))
        #print(np.sum(prob_emp))
        #prob_theo = prob_theo/np.sum(prob_theo)
        #prob_emp = prob_emp / np.sum(prob_emp)

        #loss = np.abs(prob_theo - prob_emp)
        #loss_grids = loss



        return prob_theo, prob_emp, loss_grids

    def BM_joint_dstr(self, data, time_st, delta_t, target1, target2, time1, time2):

        N=5

        count = 0
        total = 0
        s = []
        # for i in range(2):
        for j in range(data.shape[1]):
            if time_st[j] >= time1 and time_st[j] <= time2:
                # total += 1
                for i in range(data.shape[0]):
                    if data[i, j] >= target1 and data[i, j] <= target2:
                        s.append(data[i,j])




        if len(s) >= 1:
            delta_t = 10/self.Nx
            var = self.sigma**2
            len_s = len(s)
            t1 = delta_t

            dom = (np.sqrt(2*np.pi))**(len_s-1)*np.sqrt((delta_t)**(len_s-1))
            #factor_sum = ((1/self.sigma)*(np.log(s[1]/self.s0)-(self.mu-0.5*self.sigma**2)*t1))**2/t1

            factor_sum = ((1 / self.sigma) * (np.log(s[0] / self.s0))) ** 2 / t1

            for i in range(1,len(s)-1):
                    #factor_sum += ((1/self.sigma)* np.log(s[i+1]/s[i])-(1/self.sigma)*(self.mu-0.5*self.sigma**2)*delta_t)**2/delta_t
                factor_sum += ((1 / self.sigma) * np.log(s[i + 1] / s[i])-(self.mu-0.5*var)*delta_t) ** 2 / delta_t
            factor_sum = -factor_sum * 0.5

            factor = np.exp(factor_sum)
            pdf = factor / dom
        else:
            pdf = 0

        return pdf

class BM_joint:

    def __init__(self, sigma, x0):

        self.sigma = sigma
        self.x0 = x0

    def cubid_vector(self, x_range):

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x for x in x_range]
        #x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x_range

        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

    #def BM_joint_fun(self, dim, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    def BM_joint_fun(self, *args):
        delta_t = 1

        dim = 10

        const_pi = 1/(np.sqrt(2*np.pi)**dim)

        const_sigma = (1/(2*self.sigma*delta_t))

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = args

        sum_fun = (np.log(x1/self.x0))**2 + (np.log(x2/x1))**2 + (np.log(x3/x2))**2 + (np.log(x4/x3))**2 + (np.log(x5/x4))**2
        + (np.log(x6 / x5)) ** 2 + (np.log(x7/x6))**2 + (np.log(x8/x7))**2 + (np.log(x9/x8))**2 + (np.log(x10/x9))**2

        return const_pi*const_sigma*sum_fun

        #int_val = integrate.nquad(fun, [self.range_1, self.range_2, self.range_3, self.range_4, self.range_5, self.range_6, self.range_7, self.range_8, self.range_9, self.range_10])



    def cubid_prob(self, x_range):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = self.cubid_vector(x_range)

        #int_val = integrate.nquad(self.BM_joint_fun,
        #                          [self.range_1(x1), self.range_2(x2), self.range_3(x3), self.range_4(x4), self.range_5(x5),
        #                           self.range_6(x6), self.range_7(x7), self.range_8(x8), self.range_9(x9), self.range_10(x10)])
        int_val = integrate.nquad(self.BM_joint_fun,
                                  [self.range_1, self.range_2, self.range_3, self.range_4, self.range_5,
                                   self.range_6, self.range_7, self.range_8, self.range_9, self.range_10])
        return int_val

    def range_1(self, x1):

        return [0,x1]

    def range_2(self, x2):

        return [0,x2]

    def range_3(self, x3):

        return [0,x3]

    def range_4(self, x4):

        return [0,x4]

    def range_5(self, x5):

        return [0,x5]

    def range_6(self, x6):

        return [0,x6]

    def range_7(self, x7):

        return [0,x7]

    def range_8(self, x8):

        return [0,x8]

    def range_9(self, x9):

        return [0,x9]

    def range_10(self, x10):

        return [0,x10]


























