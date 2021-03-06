import mcint
import random
import math
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
import mixedvines



class MC_fdd:
    def __init__(self, sigma, x0, data, drift):


        self.sigma = sigma
        self.x0 = x0
        self.N = 100
        min_x = min(data[:,-1])
        max_x = max(data[:,-1])

        self.x_dom = [min_x, max_x]
        self.data = data
        self.drift = drift
        self.bm, self.stand_bm = self.extract_bm(self.data, self.drift)
        self.cov = self.bm_cov()



    def sum_pdf(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):


        sum_fun = (np.log(x1 / self.x0)) ** 2 + (np.log(x2 / x1)) ** 2 + (np.log(x3 / x2)) ** 2 + (
            np.log(x4 / x3)) ** 2 + (
                      np.log(x5 / x4)) ** 2
        + (np.log(x6 / x5)) ** 2 + (np.log(x7 / x6)) ** 2 + (np.log(x8 / x7)) ** 2 + (np.log(x9 / x8)) ** 2 + (
            np.log(x10 / x9)) ** 2

        return sum_fun

    def integrand(self,x):
        delta_t = 1

        dim = 10

        const_pi = 1 / (np.sqrt(2 * np.pi) ** dim)

        const_sigma = (1 / (2 * self.sigma * delta_t))

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]

        sum_part = self.sum_pdf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        return const_pi*const_sigma*sum_part

    def sampler(self):

        while True:
            x1 = random.uniform(self.x_dom[0],self.x_dom[1])
            x2 = random.uniform(self.x_dom[0], self.x_dom[1])
            x3 = random.uniform(self.x_dom[0], self.x_dom[1])
            x4 = random.uniform(self.x_dom[0], self.x_dom[1])
            x5 = random.uniform(self.x_dom[0], self.x_dom[1])
            x6 = random.uniform(self.x_dom[0], self.x_dom[1])
            x7 = random.uniform(self.x_dom[0], self.x_dom[1])
            x8 = random.uniform(self.x_dom[0], self.x_dom[1])
            x9 = random.uniform(self.x_dom[0], self.x_dom[1])
            x10 = random.uniform(self.x_dom[0], self.x_dom[1])

            yield(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

    def MC_int(self):

        result, error = mcint.integrate(self.integrand, self.sampler(), measure=1.0, n=100)

        #print("Result: ", result)
        #print("Error: ", error)

        return result, error

    def multi_mean(self, data):

        mean = np.zeros(data.shape[1])

        for i in range(data.shape[1]):
            mean[i] = np.mean(data[:,i])

        return mean


    def extract_bm(self, data, drift):

        x0_arr = [self.x0]*data.shape[1]
        log_x0_arr = np.log(x0_arr)
        bm = np.zeros_like(data)
        for i in range(data.shape[0]):
            bm[i,:] = np.log(data[i,:])-log_x0_arr

        stand_bm = self.standardized(bm, drift)

        return bm, stand_bm

    def standardized(self, data, drift):



        drift_matrix = np.ones_like(data)
        T = data.shape[0]
        #for t in range(1, T):
        #    driftT[:, t-1] = drift[t-1]*t

        for t in range(T):
            drift_matrix = data[t,:] - drift

        W = (data - drift_matrix)/self.sigma

        return W

    def multi_cov(self, data):

        return np.cov(data.T)

    def marginal_density_fun(self, inputs):


        det = np.linalg.det(self.cov)

        num_dim = self.data.shape[1]
        const = 1/(np.sqrt(2*np.pi)**(num_dim/2)**(det)**0.5)

        #x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
        #vector = np.array([self.marginal_x, x2, x3, x4, x5, x6, x7, x8, x9, x10])
        vector = []

        for i, val in enumerate(inputs):
            if i+1 == self.dim:
                vector.append(self.marginal_x)
            else:
                vector.append(val)

        vector = np.array(vector)

        exp_term = 0.5*np.dot(vector, np.linalg.inv(self.cov))
        exp_term = np.dot(exp_term, vector.T)

        pdf = const*np.exp(-exp_term)

        return pdf

    def domain(self):

        marg_len = self.data.shape[1]-1
        while True:


            sample = np.zeros(marg_len)
            for x in range(marg_len):
                sample[x] = random.uniform(self.x_dom[0], self.x_dom[1])


            yield sample

    def bm_cov(self):

        M = self.data.shape[1]-1
        cov = np.zeros((M,M))

        for i in range(2,M+2):
            for j in range(2,M+2):
                cov[i-2,j-2] = min(i,j)

        #cov = self.sigma**2*cov

        return cov

    def MC_int2(self, dim, value):

        self.dim = dim
        #dim_data = self.data[:, dim]
        self.marginal_x = value
        result, error = mcint.integrate(self.marginal_density_fun, self.domain(), measure=1.0, n=100)

        return result, error



    def empirical_marginal_pdf(self, domain, interval, cdf, sample, shifted_x):

        #p_x(x)
        #dim_data = self.data[:, dim]
        #cdf = self.cumalative_df(dim_data, N)

        idx_a = int(shifted_x / interval)

        if idx_a == 0:
            return abs(cdf[1] - cdf[idx_a])

        if idx_a == len(domain)-1:
            return abs(cdf[idx_a] - cdf[idx_a-1])


        adjust_idx = domain[idx_a] - sample
        if adjust_idx > 0:
            if adjust_idx/interval > 1:
                idx_b = idx_a - int(adjust_idx/interval)
            else:
                idx_b = idx_a - 1
        else:
            if adjust_idx / interval > 1:
                idx_b = idx_a + int(adjust_idx / interval)
            else:
                idx_b = idx_a + 1
        #idx_b = int(math.ceil((dim_data[idx]+0.1 )/ ((self.x_dom[1] - self.x_dom[0]) / N))) - 1
        #print(f"From index {idx_a} to {idx_b}.")
        #print(f"From {domain[idx_a]} with cdf {cdf[idx_a]} to {domain[idx_b]} with cdf {cdf[idx_b]}.")
        #try:
        #pdf = abs(cdf[idx_b] - cdf[idx_a])
        #except (cdf[idx_b] - cdf[idx_a]) > 0.1:

        #print(f"From index {idx_a} to {idx_b}")

        #return pdf
        return abs(cdf[idx_b] - cdf[idx_a])



    def cumalative_df(self, data, domain):

        #domain = np.linspace(min(data), max(data), N)
        N = len(domain)
        hist, bin_edges = np.histogram(data, bins=domain, density=False)


        all_sum = sum(hist)
        result = [sum(hist[:i])/all_sum for i in range(N)]


        return result

    def cdf_domain_info(self, data, dim, N):

        dim_data = data[:, dim]
        domain = np.linspace(min(dim_data), max(dim_data), N)
        domain_range = domain[-1] - domain[0]
        interval = domain_range / N

        return dim_data, domain, domain_range, interval

def check_two_distribution(data, process):

    mc = MC_fdd(process.sigma, process.s0, data, process.drift)
    num_dim = len(data[0])

    sum_error = 0
    error = np.zeros(num_dim)

    N = 5000
    num_sampling = 1000
    sum_mdf = 0
    #Measure the mdf for each dimension
    for dim in range(num_dim):
        dim_data, domain, domain_range, interval = mc.cdf_domain_info(data, dim, N)

        cdf = mc.cumalative_df(dim_data, domain)

        sampling = np.random.uniform(min(dim_data), max(dim_data), num_sampling)

        # send the sampling to compute the empirical and theoritical pdf
        for sample in sampling:
            shifted_x = np.abs(sample - min(dim_data))
            e_mdf = mc.empirical_marginal_pdf(domain, interval, cdf, sample, shifted_x)
            t_mdf, err = mc.MC_int2(dim, sample)
            sum_error += (e_mdf - t_mdf)**2
            #compute the error of 2 mdf of average
            sum_mdf += e_mdf
        error[dim] = np.sqrt(sum_error/num_sampling)
        #print(f"Average of empirical mdf at dimension {dim} is {sum_mdf/num_sampling}")
        sum_error = 0
        sum_mdf = 0


    return error






class dim_reduction:



    def principle_component(self, data):

        x = self.data_standardize(data)

        pca = PCA(n_components=2)

        principalComponents = pca.fit_transform(x)

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        #finalDf = pd.concat([principalDf, df[['target']]], axis=1)

        return principalComponents, principalDf




    def data_standardize(self, data):

        data_scaled = StandardScaler().fit_transform(data)

        features = data_scaled


        return features

    def visualize(self, pca_components, pca_df):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        #targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        #colors = ['r', 'g', 'b']

        for comp in pca_components:
            #indicesToKeep = finalDf['target'] == target
            ax.scatter(comp[0]
                       , comp[1])
        ax.legend()
        ax.grid()

        plt.show()


