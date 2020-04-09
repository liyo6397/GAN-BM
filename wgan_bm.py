from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stoch_process import Geometric_BM
import scipy.stats as stats


# defining functions for the two networks.
# Both the networks have two hidden layers
# and an output layer which are densely or
# fully connected layers defining the
# Generator network function

class WGAN():

    def __init__(self, paths, method,batch_size):

        # Input
        self.ini_path = paths[:,0]
        self.paths = paths
        self.N = paths.shape[0]
        self.unknown_days = paths.shape[1]
        self.noise_method = method

        # GAN
        self.num_units = 10
        self.paths_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])
        self.z_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])
        self.batch_size = batch_size
        self.n_batch = self.paths.shape[0] % self.batch_size

        # G & D
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.G_sample = self.generator(self.z_tf)
        self.D_output_real = self.discriminator(self.paths)
        self.D_output_fake = self.discriminator(self.G_sample)


        # Loss function
        #self.g_samp = tf.zeros(shape=[self.N, self.paths.shape[1]], dtype=tf.float64)
        #self.g_samp = np.zeros((self.N, self.paths.shape[1]))

        #self.D_loss = tf.reduce_mean(self.D_output_real) - tf.reduce_mean(self.D_output_fake)+self.lip_sum
        self.D_loss = tf.reduce_mean(self.D_output_real) - tf.reduce_mean(self.D_output_fake)
        #self.ini_cond_loss = tf.reduce_mean(tf.sqrt(tf.square(self.G_sample[:,0]-self.ini_path)))
        self.G_loss = -tf.reduce_mean(self.D_output_fake)
        #self.D_group_loss, self.G_group_loss = self.group_loss()


        # Optimizing
        self.D_solver, self.G_solver, self.clip_D = self.optimizer()

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

    def Homo_Poisson(self,m,n):

        rand_poi = np.random.poisson(0.1,size=[m,n])

    def group_Loss(self):

        for idx in range(unknown_days):
            Simu = Simulation(idx_elment=idx)

            new_samples = Simu.collect_pts(paths_pred)
            wgan_samples[str(idx)] = new_samples

            original_samples = Simu.collect_pts(paths)
            o_samples[str(idx)] = original_samples

    def generator(self, z):

        with tf.variable_scope("gen", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs=z, units=self.num_units,
                                     activation=tf.nn.leaky_relu)

            # for l in range(self.layers-2):
            #    hidden = tf.layers.dense(inputs=hidden, units=self.num_units,
            #                             activation=tf.nn.leaky_relu)

            output = tf.layers.dense(inputs=hidden, units=self.paths.shape[1])


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

    def optimizer(self):

        # Select parameters
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]

        clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in disc_vars]

        # Optimizers
        D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(-self.D_loss, var_list=disc_vars)
        #Gini_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.ini_cond_loss, var_list=gen_vars)
        G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss, var_list=gen_vars)


        return D_solver, G_solver, clip_D #, Gini_solver

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

    def train_nbatch2(self,itrs,print_itr):

        for it in range(itrs):
            #paths_batch = self.create_mini_batches(self.paths,self.batch_size)
            # random sampling
            gbm = Geometric_BM(self.paths.shape[0], self.unknown_days-1, 0, 0.1)
            paths_batch = gbm.predict_path()

            tf_dict = {self.paths_tf: paths_batch,
                       self.z_tf: self.sample_Z(paths_batch.shape[0], paths_batch.shape[1])}

            # Run disciminator solver
            for it_d in range(5):
                _, D_loss_curr, _ = self.sess.run([self.D_solver, self.D_loss, self.clip_D], tf_dict)

            # Run generator solver
            #tf_dict = {self.paths_tf: paths_batch,
            #               self.z_tf: self.sample_Z(paths_batch.shape[0], paths_batch.shape[1])}
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)

            # Print loss
            if it % print_itr == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))

            if it == itrs-1:
                samples = self.sess.run(self.G_sample, feed_dict={self.z_tf: self.sample_Z(self.paths.shape[0], self.paths.shape[1])})



        return samples


    def train_nbatch(self,itrs,print_itr):

        dg_loss = 1
        d_loss = 1
        it = 1

        while d_loss >= 10**(-7):
        #for it in range(itrs):
            #paths_batch = self.create_mini_batches(self.paths,self.batch_size)
            # random sampling
            rand_index = np.random.choice(self.paths.shape[0], size=self.batch_size)
            random_batches = self.paths[rand_index, :]
            mini_batch_size = int(self.batch_size/10)
            mini_batches = self.create_mini_batches(random_batches, mini_batch_size)

            for paths_batch in mini_batches:
                tf_dict = {self.paths_tf: paths_batch,
                           self.z_tf: self.sample_Z(paths_batch.shape[0], paths_batch.shape[1])}

                # Run disciminator solver
                for it_d in range(5):
                    _, D_loss_curr, _ = self.sess.run([self.D_solver, self.D_loss, self.clip_D], tf_dict)


                # Run generator solver
                #tf_dict = {self.paths_tf: paths_batch,
                #               self.z_tf: self.sample_Z(paths_batch.shape[0], paths_batch.shape[1])}
                _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)

                # Print loss

            if it % print_itr == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))

            #if it == itrs-1:
                #samples = self.sess.run(self.G_sample, feed_dict={self.z_tf: self.sample_Z(self.paths.shape[0], self.paths.shape[1])})
                #num_pts = self.batch_size*10
            samples = self.sess.run(self.G_sample,
                                            feed_dict={self.z_tf: self.sample_Z(self.paths.shape[0], self.paths.shape[1])})
            it += 1
            d_loss = np.square(D_loss_curr)
            #g_loss = np.square(G_loss_curr)

            #dg_loss = np.sqrt(d_loss + g_loss)

            if it >= 5 * 10 ** 4:
                break

        return samples



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
            #_, ini_loss_curr = self.sess.run([self.Gini_solver, self.ini_cond_loss], tf_dict)
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)



            self.g_samp = self.sess.run(self.G_sample, feed_dict={self.z_tf: self.sample_Z(100, self.paths.shape[1])})



            #DG_loss = np.abs(D_loss_curr - G_loss_curr)
            #D_loss = np.abs(D_loss_curr)

            d_loss = np.square(D_loss_curr)
            g_loss = np.square(G_loss_curr)
            dg_loss = np.sqrt(d_loss+g_loss)

            it += 1
            # Print loss
            if it % print_itr == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))
                ganLoss_d.append(D_loss_curr)
                ganLoss_g.append(G_loss_curr)

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

    def __init__(self,save):

        self.save = save

    def loss_plot(self,loss_d,loss_g):

        plt.plot(loss_d,label="Discriminator Loss")
        plt.plot(loss_g, label="Generator Loss")
        plt.xlabel("Number of 1000 Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(np.abs(loss_d), label="Discriminator Loss")
        plt.plot(np.abs(loss_g), label="Generator Loss")
        plt.xlabel("Number of 1000 Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def plot_dstr_set(self, gan_samp, org_samp, n_ipts,s0,mu,sigma):

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        paths_org = np.array([org_samp[str(scen)] for scen in range(n_ipts)])

        #mu = 0
        #sigma = 0.1
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        for t in range(1,n_ipts):

            path = paths[t, :]
            mean = s0 * np.exp((mu - sigma ** 2 / 2) * t)
            var = s0 ** 2 * np.exp(2 * mu * t + t * sigma ** 2) * (np.exp(t * sigma ** 2) - 1)
            std = np.sqrt(var)

            xmin = stats.lognorm(std, scale=np.exp(mean)).ppf(0.001)
            xmax = stats.lognorm(std, scale=np.exp(mean)).ppf(0.999)


            x = np.linspace(xmin, xmax, 10000)
            pdf = (np.exp(-(np.log(x) - mean) ** 2 / (2 * std ** 2))/ (x * std * np.sqrt(2 * np.pi)))
            #pdf = (np.exp(-(np.log(x) - (mu - sigma ** 2 / 2) * t) ** 2 / (2 * t * sigma ** 2)) / (x * std * np.sqrt(2 * np.pi*t)))


            ax = fig.add_subplot(4, 3, t)
            ax.plot(x, pdf, 'k', label="Geometric Brownian Motion")
            ax.hist(path, 20, density=True)


            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")
            if self.save:
                plt.savefig("images/dstr_wgan_{}1000.png".format(str(method)), bbox_inches='tight')

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

        for t in range(1,n_ipts):

            path_gan = paths_gan[t, :]
            path_org = paths_org[t, :]



            ax = fig.add_subplot(4, 3, t)
            ax.hist(path_org, 20, density=True, label="Geometric Brownian Motion")
            ax.hist(path_gan, 20, density=True, label="WGAN Sampling")


            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")
            if self.save:
                plt.savefig("images/dstr_wgan_hist{}.png".format(str(method)), bbox_inches='tight')

        plt.show()

    def plot_2path(self,gbm_paths,model_paths,method,s0):

        fig = plt.figure(figsize=(8, 3))
        #fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        paths = gbm_paths
        for k in range(2):
            if k ==0:
                ax = fig.add_subplot(1, 2, k+1)
            else:
                ax = fig.add_subplot(1, 2, k + 1, sharex=ax,sharey=ax)

            for i in range(paths.shape[0]):
                ax.plot(paths[i, :])
            if k == 0:
                ax.title.set_text("Geometric Brownian Motion")
            else:
                ax.title.set_text("WGAN-{}".format(str(method)))
            #paths = model_paths
            paths = np.hstack((np.array([[s0] for scen in range(model_paths.shape[0])]), model_paths))





        if self.save:
            fig.savefig("images/path_wgan_{}1000.png".format(str(method)), bbox_inches='tight')
        plt.show()



if __name__ == "__main__":

    # BM
    number_inputs = 100
    unknown_days = 11
    mu = 0
    sigma = 0.1
    method = "uniform"
    converge_crit = 10**(-6)
    print_itr = 1000
    batch_size = 10
    save = False

    gbm = Geometric_BM(number_inputs, unknown_days, mu, sigma)
    graph = Plot_result(save)
    paths = gbm.predict_path()
    s0 = gbm.s0

    # GAN-distribution
    wgan = WGAN(paths[1:,:], method,batch_size)
    paths_pred, loss_d, loss_g =wgan.train(converge_crit, print_itr)

    Sim = Simulation(unknown_days, paths, paths_pred)
    r_samples, g_samples = Sim.samples()

    # Plot GAN
    graph.loss_plot(loss_d,loss_g)
    graph.plot_dstr_set(g_samples, r_samples, unknown_days,s0,mu,sigma)
    graph.plot_dstr_set_hist(g_samples, r_samples, unknown_days, s0, mu, sigma)
    graph.plot_2path(paths, paths_pred,method,s0)




