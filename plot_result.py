import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import ticker, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class plot_result():

    def __init__(self,save, data_type, method):

        self.save = save
        self.data_type = data_type
        self.method = method

    def loss_plot(self,loss_d,loss_g):

        plt.plot(loss_d,label="Discriminator Loss")
        plt.plot(loss_g, label="Generator Loss")
        plt.xlabel("Number of 1000 Iterations")
        plt.ylabel("Loss")
        plt.legend()
        if self.save:
            plt.savefig("slides/images/loss/loss_wgan_{}".format(str(self.method)), bbox_inches='tight')
        plt.show()


    def dstr_ou(self,gan_samp, n_ipts, s0, sigma, theta, t0, tend):

        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        ts = np.linspace(t0, tend, n_ipts)

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)


        for i in range(1, n_ipts):

            path = paths[i, :]
            t = ts[i]
            #mean = s0*np.exp(-theta*t)
            mean = s0*np.exp(theta*t)

            #var = sigma**2 * (1-np.exp(-2*theta*t))/(2*theta)
            var = sigma ** 2 * (np.exp(2 * theta * t)-1) / (2 * theta)
            std = np.sqrt(var)

            xmin = stats.norm(loc=mean, scale=std).ppf(0.001)
            xmax = stats.norm(loc=mean, scale=std).ppf(0.999)
            #xmin = stats.norm.ppf(0.001)
            #xmax = stats.norm.ppf(0.999)

            x = np.linspace(xmin, xmax, 10000)
            term1 = 1/(std*np.sqrt(2*np.pi))
            term2 = np.exp(-0.5*((x-mean)/std)**2)
            pdf = term1*term2
            ax = fig.add_subplot(4, 3, i)
            ax.plot(x, pdf, 'k', label=self.data_type)
            #ax.plot(x, stats.norm.pdf(x), 'k-', lw=2)
            ax.hist(path, 20, density=True)

            ax.title.set_text("t = %2f" % (t))
            ax.set(xlabel="x", ylabel="PDF")
            if self.save:
                plt.savefig("slides/images/distribution_bm/dstr_wgan_{}".format(str(self.method)), bbox_inches='tight')

        plt.show()




    def dstr_gbm(self, gan_samp, n_ipts,s0,mu,sigma):


        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])



        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        for t in range(1,n_ipts):

            path = paths[t, :]
            mean = np.log(s0) + mu*t
            var = sigma*np.sqrt(t)
            std = np.sqrt(var)

            xmin = stats.lognorm(std, scale=np.exp(mean)).ppf(0.001)
            xmax = stats.lognorm(std, scale=np.exp(mean)).ppf(0.999)

            x = np.linspace(xmin, xmax, 10000)
            #pdf = (np.exp(-(np.log(x) - mean) ** 2 / (2 * std ** 2))/ (x * std * np.sqrt(2 * np.pi)))
            pdf = np.exp(-0.5*((np.log(x)-np.log(s0)-mu*t)/(sigma*np.sqrt(t)))**2) / (x * sigma * np.sqrt(2 * np.pi*t))


            ax = fig.add_subplot(4, 3, t)
            ax.plot(x, pdf, 'k', label=self.data_type)
            ax.hist(path, 20, density=True)


            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")
            if self.save:
                plt.savefig("slides/images/distribution_bm/dstr_wgan_{}".format(str(self.method)), bbox_inches='tight')

        plt.show()

    def plot_dstr_set_hist(self, gan_samp, org_samp, n_ipts):

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        paths_gan = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        paths_org = np.array([org_samp[str(scen)] for scen in range(n_ipts)])

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        for t in range(1,n_ipts-1):

            path_gan = paths_gan[t, :]
            path_org = paths_org[t, :]



            ax = fig.add_subplot(3, 3, t)
            ax.hist(path_org, 20, density=True, label=self.data_type)
            ax.hist(path_gan, 20, density=True, label="WGAN Sampling")


            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")



        plt.show()

    def plot_2path(self,gbm_paths,model_paths,method,s0):

        fig = plt.figure(figsize=(8, 3))

        paths = gbm_paths
        for k in range(2):
            if k ==0:
                ax = fig.add_subplot(1, 2, k+1)
            else:
                ax = fig.add_subplot(1, 2, k + 1, sharex=ax,sharey=ax)

            for i in range(paths.shape[0]):
                ax.plot(paths[i, :])
            if k == 0:
                ax.title.set_text(self.data_type)
            else:
                ax.title.set_text("WGAN-{}".format(str(method)))
            #paths = model_paths
            paths = np.hstack((np.array([[s0] for scen in range(model_paths.shape[0])]), model_paths))


        if self.save:
            fig.savefig("slides/images/path_bm/path_wgan_{}.png".format(str(method)), bbox_inches='tight')
        plt.show()

    def fdd_plot(self, data_1, data_2, time_st, loss):

        Nx = loss.shape[0]
        X = np.linspace(0, time_st[-1], Nx)

        min_data1 = min(data_1[:, -1])
        min_data2 = min(data_2[:, -1])
        max_data1 = max(data_1[:, -1])
        max_data2 = max(data_2[:, -1])

        Y = np.linspace(min(min_data1, min_data2), max(max_data1, max_data2), Nx)


        #fig, ax = plt.subplots()
        #cs = ax.contourf(X,Y,loss, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
        cs = plt.contourf(X, Y, loss, locator = ticker.LogLocator())
        plt.colorbar(cs)

        plt.show()

    def fdd_plot_3D(self, time_st, data1, data2, den_1, den_2, loss):

        X = np.linspace(time_st[0], time_st[-1], den_1.shape[0])
        min_data1 = min(map(min, data1))
        min_data2 = min(map(min, data2))
        max_data1 = max(map(max, data1))
        max_data2 = max(map(max, data2))
        Y1 = np.linspace(min_data1, max_data1, den_1.shape[1])
        Y2 = np.linspace(min_data2, max_data2, den_1.shape[1])

        fig, axs = plt.subplots(ncols=2, figsize=(8, 3))

        #plt.figure(figsize=(8, 3))
        #plt.subplot(1, 2, 1)

        den = den_1
        Y = Y1
        count = 1
        for ax in axs:
            if count == 1:
                ax.contourf(X, Y, den, 20)
                ax.title.set_text("Ornstein-Uhlenbeck process")
            else:
                ax.contourf(X, Y, den, 10)
                ax.title.set_text("WGAN")
            contours = ax.contour(X, Y, den, 10, colors='black')
            plt.clabel(contours, inline=True, fontsize=8)
            ax.grid()
            den = den_2
            count += 1
            ax1 = ax

        #fig.colorbar(contf, ax1)
        plt.show()




        '''plt.contourf(X,Y1,den_1, 10)
        plt.colorbar()
        contours = plt.contour(X, Y1, den_1, 10, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        #plt.imshow(den_1, extent=[X[0], X[-1], 0, 1.5], origin='lower',
        #           cmap='RdGy', alpha=0.5)


        plt.title("Joint Probability Density of Geometric Brownian Motion")
        plt.xlabel("Time")
        plt.ylabel("The value of process")
        plt.legend()
        plt.grid()
        plt.show()


        plt.contourf(X,Y2,den_2, 10)
        plt.colorbar()
        contours = plt.contour(X, Y2, den_2, 10, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        #plt.imshow(den_2, extent=[X[0], X[-1], Y2[0], Y2[-1]], origin='lower',
        #           cmap='RdGy', alpha=0.5)

        plt.title("Joint Probability Density of WGAN Simulation")
        plt.xlabel("Time")
        plt.ylabel("The value of process")
        plt.legend()
        plt.grid()
        plt.show()'''

        plt.contourf(X, Y1, loss, 10)
        plt.colorbar()
        contours = plt.contour(X, Y1, loss, 10, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        #plt.imshow(loss, extent=[X[0], X[-1], Y1[0], Y1[-1]], origin='lower',
        #           cmap='RdGy', alpha=0.5)
        plt.title("Loss Plot for Joint Probability Density")
        plt.xlabel("Time")
        plt.ylabel("The value of process")
        plt.legend()
        plt.grid()
        plt.show()

    def fdd_plot_scatter(self, data, prob_theo, prob_emp, time_st, loss):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        N = len(prob_theo)
        #theoritical
        xs = np.linspace(0,data.shape[1], N)
        ys = np.linspace(min(data[:,-1]),max(data[:,-1]), N)

        ax.scatter(xs, ys, prob_theo, marker='o')
        ax.scatter(xs, ys, prob_emp, marker='v')

        #fig.colorbar(p, shrink=0.5, aspect=5)

        plt.show()

    def fdd_plot_2Dscatter(self, data, prob_theo, prob_emp, process):
        fig = plt.figure()


        N = len(prob_theo)
        #theoritical

        x = np.linspace(min(data[:,-1]),max(data[:,-1]), N)

        plt.plot(x, prob_theo, 'r', label="gbm")
        plt.plot(x, prob_emp, 'b', label=process)
        plt.xlabel("Values of {}".format(process))
        plt.ylabel("Probability density")
        plt.legend()

        #fig.colorbar(p, shrink=0.5, aspect=5)

        plt.show()

