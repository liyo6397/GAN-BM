import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
            mean = s0*np.exp(-theta*t)

            var = sigma**2 * (1-np.exp(-2*theta*t))/(2*theta)
            std = np.sqrt(var)

            #xmin = stats.norm(std, scale=mean).ppf(0.001)
            #xmax = stats.norm(std, scale=mean).ppf(0.999)
            xmin = stats.norm(loc=mean, scale=std).ppf(0.001)
            xmax = stats.norm(loc=mean, scale=std).ppf(0.999)
            xmin = stats.norm.ppf(0.001)
            xmax = stats.norm.ppf(0.999)

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

    def plot_dstr_set_hist(self, gan_samp, org_samp, n_ipts,s0,mu,sigma):

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

    def fdd_plot(self,X,Y,loss):

        plt.contourf(X,Y,loss)
        plt.colorbar()
        plt.show()
        '''plt.imshow(loss, extent=[0, loss.shape[0], 0, loss.shape[0]], origin='lower',
                   cmap='RdGy')
        plt.colorbar()
        plt.axis(aspect='image')'''
