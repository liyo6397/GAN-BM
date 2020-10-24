import mcint
import random
import math
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd


class MC_fdd:
    def __init__(self, sigma, x0, x_range):


        self.sigma = sigma
        self.x0 = x0
        self.x_dom = x_range

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

        print("Result: ", result)
        print("Error: ", error)

    def multi_mean(self, data):

        mean = np.zeros(data.shape[1])

        for i in range(data.shape[1]):
            mean[i] = np.mean(data[:,i])

        return mean

    def multi_cov(self, data):

        return np.cov(data)

    def extract_bm(self, data, drift):

        x0_arr = [self.x0]*data.shape[1]
        log_x0_arr = np.log(x0_arr)
        bm = np.zeros_like(data)
        for i in range(data.shape[0]):
            bm[i,:] = np.log(data[i,:])-log_x0_arr

        stand_bm = self.standardized(bm, drift)

        return bm, stand_bm

    def standardized(self, data, drift):



        driftT = np.ones_like(data)
        T = data.shape[1]
        for t in range(1, T):
            driftT[:, t-1] = drift[t-1]*t

        W = (data - driftT)/self.sigma

        return W



















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



if __name__ == '__main__':
    sigma = 0.1
    x0 = 0.1
    x_range = [0.1, 0.5]

    MC = MC_fdd(sigma, x0, x_range)
    MC.MC_int()


