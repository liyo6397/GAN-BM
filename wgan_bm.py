from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM, Orn_Uh
import scipy.stats as stats


# defining functions for the two networks.
# Both the networks have two hidden layers
# and an output layer which are densely or
# fully connected layers defining the
# Generator network function

class WGAN():

    def __init__(self, paths, method):

        # Input
        self.ini_path = paths[:,0]
        self.paths = paths
        self.N = paths.shape[0]
        self.unknown_days = paths.shape[1]
        self.noise_method = method

        # GAN
        self.num_units = paths.shape[1]
        self.paths_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])
        self.z_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])

        # G & D
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.G_sample = self.generator(self.z_tf, paths.shape[1])
        self.D_output_real = self.discriminator(self.paths)
        self.D_output_fake = self.discriminator(self.G_sample)


        self.D_loss = tf.reduce_mean(self.D_output_real) - tf.reduce_mean(self.D_output_fake)
        self.G_loss = -tf.reduce_mean(self.D_output_fake)
        self.DG_loss = tf.sqrt(tf.square(self.D_loss) + tf.square(self.G_loss))

        self.disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
        self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(-self.D_loss, var_list=self.disc_vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss, var_list=self.gen_vars)
        self.DG_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.DG_loss)
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.disc_vars]

        # Optimizing
        #self.D_solver, self.G_solver, self.clip_D = self.optimizer()


        self.sess.run(tf.global_variables_initializer())

    def sample_Z(self, m, n):

        # random distribution
        if self.noise_method =='uniform':
        #z_process= self.random_distribution(m,n)
            z_process = np.random.uniform(-1, 1, size=[m, n])
        if self.noise_method == 'normal':
            z_process = np.random.normal(0, 1, size=[m, n])

        if self.noise_method == 'brownian':
            z_process = self.Brownian(m,n)

        return z_process


    def Brownian(self,m,n):

        delta = 0.2
        dt = n/(n-1)

        b = {str(scen): np.random.normal(0, scale=delta*dt, size = int(n)) for scen in range(1, m + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, m + 1)}

        W_process = np.array([W[str(scen)] for scen in range(1, m + 1)])

        return W_process


    def generator(self, z, output_units):

        with tf.variable_scope("gen", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs=z, units=self.num_units,
                                     activation=tf.nn.leaky_relu)

            # for l in range(self.layers-2):
            #    hidden = tf.layers.dense(inputs=hidden, units=self.num_units,
            #                             activation=tf.nn.leaky_relu)

            output = tf.layers.dense(inputs=hidden, units=output_units) # change from path.shape[1]


            return output

    def discriminator(self, z):

        with tf.variable_scope("disc", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs=z, units=self.num_units,
                                     activation=tf.nn.relu)

            # for l in range(self.layers - 2):
            # hidden = tf.layers.dense(inputs=hidden, units=self.num_units,
            #                             activation=tf.nn.leaky_relu)

            output = tf.layers.dense(hidden, units=1)


            return output

    def optimizer_disc(self):

        # Select parameters
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]



        # Optimizers
        D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(-self.D_loss, var_list=disc_vars)



        return disc_vars, D_solver

    def optimizer_gen(self):

        # Select parameters
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]
        G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss, var_list=gen_vars)


        return gen_vars, G_solver


    def create_mini_batches(self,data,b_size):
        mini_batches = []
        np.random.shuffle(data)
        n_minibatches = b_size
        i = 0

        for i in range(n_minibatches):
            if data.shape[0] % b_size != 0:
                mini_batch = data[i * b_size:data.shape[0]]
                mini_batches.append(mini_batch)
            else:
                mini_batch = data[i * b_size:(i + 1) * b_size, :]
                mini_batches.append(mini_batch)

        return mini_batches


    def train(self, converg_crit,print_itr):

        dg_loss = 1
        D_loss = 1
        DG_loss = 1
        ini_loss_curr = 1
        it = 1
        ganLoss_d = []
        ganLoss_g = []


        #while ((ini_loss_curr >= converg_crit ) or (it >= 80000 and dg_loss >= converg_crit * 10 ** (1))):
        while (dg_loss >= converg_crit or (it >= 100000 and dg_loss >= converg_crit*10**(2))):

            tf_dict = {self.paths_tf: self.paths, self.z_tf: self.sample_Z(self.N, self.paths.shape[1])}



                # Run disciminator solver
            for it_d in range(5):
                _, D_loss_curr,_ = self.sess.run([self.D_solver, self.D_loss,self.clip_D], tf_dict)

            # Run generator solver
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)

            #d_loss = np.square(D_loss_curr)
            #g_loss = np.square(G_loss_curr)
            #dg_loss = np.sqrt(d_loss+g_loss)

            dg_loss, _ = self.sess.run([self.DG_loss, self.DG_solver], tf_dict)



            it += 1
            # Print loss
            if it % print_itr == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))
                ganLoss_d.append(D_loss_curr)
                ganLoss_g.append(G_loss_curr)

            self.g_samp = self.sess.run(self.G_sample, feed_dict={self.z_tf: self.sample_Z(100, self.paths.shape[1])})

        print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))
        ganLoss_d.append(D_loss_curr)
        ganLoss_g.append(G_loss_curr)





        return self.g_samp, np.array(ganLoss_d), np.array(ganLoss_g)

    def predict(self, paths):

        tf_dict = {self.paths_tf: paths, self.z_tf: self.sample_Z(paths.shape[0], paths.shape[1])}

        new_samples = self.sess.run(self.G_sample, tf_dict)

        return new_samples


class Simulation():

    def __init__(self, unknown_days, paths, pred_paths):
        self.unknown_days = unknown_days
        self.r_paths = paths
        self.g_paths = pred_paths

    def collect_pts(self, paths,idx):
        sample_dstr = []

        for vec in paths:
            sample_dstr.append(vec[idx])

        sample_dstr = np.array(sample_dstr)

        return sample_dstr


    def samples(self):
        g_samples = {}
        r_samples = {}
        for idx in range(self.unknown_days):
            #Simu = Simulation(idx_elment=idx)

            new_samples = self.collect_pts(self.g_paths,idx)
            g_samples[str(idx)] = new_samples

            real_samples = self.collect_pts(self.r_paths,idx)
            r_samples[str(idx)] = real_samples

        return r_samples, g_samples

class Plot_result():

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


    def plot_dstr_set_orn(self,gan_samp, org_samp, n_ipts,s0,mu,sigma, theta):

        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        for t in range(1, n_ipts):

            path = paths[t, :]
            mean = np.log(s0) + mu * t
            var = sigma * np.sqrt(t)
            std = np.sqrt(var)

            xmin = stats.lognorm(std, scale=np.exp(mean)).ppf(0.001)
            xmax = stats.lognorm(std, scale=np.exp(mean)).ppf(0.999)

            x = np.linspace(xmin, xmax, 10000)
            term_1 = np.sqrt(theta/(1-2*np.exp(-2*theta*t)))
            term_2 = 1/(sigma*np.sqrt(np.pi))
            term_3 = -theta/(1-2*np.exp(-2*theta*t))*((x-s0-mu*t)/sigma)**2
            pdf = term_1*term_2*np.exp(term_3)
            ax = fig.add_subplot(4, 3, t)
            ax.plot(x, pdf, 'k', label=self.data_type)
            ax.hist(path, 20, density=True)

            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")
            if self.save:
                plt.savefig("slides/images/distribution_bm/dstr_wgan_{}".format(str(method)), bbox_inches='tight')

        plt.show()




    def plot_dstr_set(self, gan_samp, org_samp, n_ipts,s0,mu,sigma):


        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        paths_org = np.array([org_samp[str(scen)] for scen in range(n_ipts)])


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
                plt.savefig("slides/images/distribution_bm/dstr_wgan_{}".format(str(method)), bbox_inches='tight')

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

        plt.contour(X, Y, loss)
        plt.colorbar()
        plt.show()



