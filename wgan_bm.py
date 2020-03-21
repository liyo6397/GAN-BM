import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from stoch_process import Geometric_BM
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing


# defining functions for the two networks.
# Both the networks have two hidden layers
# and an output layer which are densely or
# fully connected layers defining the
# Generator network function

class WGAN():

    def __init__(self, paths, number_inputs,method):

        # Input
        self.paths = paths
        self.N = paths.shape[0]
        self.unknown_days = paths.shape[1]
        self.noise_method = method

        # GAN
        self.num_units = 10
        self.paths_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])
        self.z_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])

        # G & D
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.G_sample = self.generator(self.z_tf)
        self.D_output_real = self.discriminator(self.paths)
        self.D_output_fake = self.discriminator(self.G_sample)


        # Loss function
        #self.g_samp = tf.zeros(shape=[self.N, self.paths.shape[1]], dtype=tf.float64)
        self.g_samp = np.zeros((self.N, self.paths.shape[1]))
        self.lip_sum = self.lip_loss()
        #self.D_loss = tf.reduce_mean(self.D_output_real) - tf.reduce_mean(self.D_output_fake)+self.lip_sum
        self.D_loss = tf.reduce_mean(self.D_output_real) - tf.reduce_mean(self.D_output_fake)
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
            z_process = self.Brownian()

        return z_process


    def random_distribution(self, m, n):

        return np.random.uniform(0, 1, size=[m, n])
        # np.random.normal(-1, 1, int(self.N))
        #return np.random.normal(0, 1, size=[m, n])

    def Brownian(self):

        delta = 0.2
        dt = self.unknown_days/(self.unknown_days-1)

        b = {str(scen): np.random.normal(0, scale=delta*dt, size = int(self.unknown_days)) for scen in range(1, self.N + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, self.N + 1)}

        W_process = np.array([W[str(scen)] for scen in range(1, self.N + 1)])

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
        G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss, var_list=gen_vars)

        return D_solver, G_solver, clip_D

    def norm2(self,distance):

        dis_sum = np.zeros(self.N)

        for i, dis in enumerate(distance):
            dis_sum[i] = np.sum(dis)

        max_sum = np.max(np.abs(dis_sum))

        return max_sum


    def lip_loss(self):


        #self.paths_tf = self.sess.run(self.paths_tf,feed_dict={self.paths_tf: paths})

        v =0.01

        distance = self.paths-self.g_samp
        norm2_dis = np.linalg.norm(distance,2)

        #norm2_dis = self.norm2(distance)

        lip = np.abs(np.abs(self.D_output_real-self.D_output_fake)/tf.abs(norm2_dis)-1.)

        lip_sum = tf.reduce_sum(lip)

        lip = v*lip_sum


        #D_loss = -tf.reduce_mean(self.D_output_real) + tf.reduce_mean(self.D_output_fake)+lip
        #G_loss = -tf.reduce_mean(self.D_output_fake)

        return lip



    def train(self, itrs):

        tf_dict = {self.paths_tf: self.paths, self.z_tf: self.sample_Z(self.N, self.paths.shape[1])}

        start_time = time.time()
        for it in range(itrs):
            #if it == itrs - 1:
            self.g_samp = self.sess.run(self.G_sample, tf_dict)

            for it_d in range(5):
                # Run disciminator solver
                _, D_loss_curr,_ = self.sess.run([self.D_solver, self.D_loss,self.clip_D], tf_dict)

            # Run generator solver
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)

            # Print loss
            if it % 1000 == 0:
                # if it % 100 == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))
                # if it == itrs-1:
                #    print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))

        return self.g_samp

    def predict(self, paths):

        tf_dict = {self.paths_tf: paths, self.z_tf: self.sample_Z(paths.shape[0], paths.shape[1])}

        new_samples = self.sess.run(self.G_sample, tf_dict)

        return new_samples

    def predict_path(self, paths):

        # tf_dict = {self.paths_tf: paths, self.z_tf: self.sample_Z(self.N,self.paths.shape[1])}
        tf_dict = {self.z_tf: self.sample_Z(self.N, self.paths.shape[1])}

        new_samples = self.sess.run(self.G_sample, tf_dict)

        return new_samples


class Simulation():

    def __init__(self, idx_elment):
        self.idx_elm = idx_elment - 1

    def collect_pts(self, paths):
        sample_dstr = []

        for vec in paths:
            sample_dstr.append(vec[self.idx_elm])

        sample_dstr = np.array(sample_dstr)

        return sample_dstr

    def path_lognormal_dstr(self, samples):
        mean = np.mean(samples)
        std = np.std(samples)
        new_samples = np.random.lognormal(mean, std, len(samples))

        return new_samples


class Plot_result():

    def plot_dstr(self, original_samples, new_samples, idx):
        # Plotting the simulations

        plot1 = plt.figure(1)
        plt.title("Geometric Brownian Motion-{}th day".format(str(idx)))
        plt.hist(original_samples)
        # plt.savefig('images/out/gbm_{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        # plt.show()

        plot2 = plt.figure(2)
        plt.title("Samples generated by GAN-{}th day".format(str(idx)))
        plt.hist(new_samples)
        # plt.savefig('images/distribution/gan_{}.png'.format(str(idx)))
        # plt.savefig('images/out/gan_{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        plt.show()

    def z_value(self, path_gan, path_org):

        mu_g = np.mean(path_gan)

        var_g = np.std(path_gan) / np.sqrt(len(path_gan))

        # s, mu_o, scale = stats.lognorm.fit(path_org, floc=0)
        # var_o = s/np.sqrt(len(path_org))
        mu_o = np.mean(path_org)
        sigma_o = np.std(path_org) / np.sqrt(len(path_org))

        z = (mu_g - mu_o) / (np.sqrt(var_g + sigma_o ** 2))

        return z

    def normalize(self, data):

        xmin = data.min()
        xmax = data.max()
        data_range = xmax - xmin

        for i, val in enumerate(data):
            data[i] = (val - xmin) / data_range

        return data

    def plot_dstr_set(self, gan_samp, org_samp, n_ipts,s0,save):

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        paths_org = np.array([org_samp[str(scen)] for scen in range(n_ipts)])

        # plt.text(0, 0.1, r"The distribution of noise data: Brownian Motion")
        mu = 0
        sigma = 0.1
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        for t in range(1,n_ipts):
            #plt.title(
            #    "Data distribution:\n Geometric Brownian Motion and generator sampling-{}th day".format(str(scen)))
            path = paths[t, :]
            mean = s0 * np.exp((mu + sigma ** 2 / 2) * t)
            var = s0 ** 2 * np.exp(2 * mu * t + t * sigma ** 2) * (np.exp(t * sigma ** 2) - 1)
            std = np.sqrt(var)

            xmin = stats.lognorm(std, scale=np.exp(mean)).ppf(0.001)
            xmax = stats.lognorm(std, scale=np.exp(mean)).ppf(0.999)


            x = np.linspace(xmin, xmax, 10000)
            pdf = (np.exp(-(np.log(x) - mean) ** 2 / (2 * std ** 2))/ (x * std * np.sqrt(2 * np.pi)))

            #plt.plot(x, pdf, 'k', label="Geometric Brownian Motion")
            #plt.hist(path, 50, density=True, stacked=True)
            ax = fig.add_subplot(4, 3, t)
            ax.plot(x, pdf, 'k', label="Geometric Brownian Motion")
            ax.hist(path, 20, density=True)


            ax.title.set_text("{}th day".format(str(t)))
            ax.set(xlabel="x", ylabel="PDF")
            if save:
                plt.savefig("images/dstr_wgan_{}1000.png".format(str(method)), bbox_inches='tight')

        plt.show()

    def aderson_test(self, g_sample, s, mu, scale):

        size_g = len(g_sample)
        g_sample.sort()
        sum_ad = 0

        for i in range(size_g):
            F_c = stats.lognorm(s=s, loc=mu, scale=scale).cdf(g_sample[i])
            F_a = stats.lognorm(s=s, loc=mu, scale=scale).cdf(g_sample[size_g - i - 1])

            sum_ad += (2 * (i + 1) - 1) * (np.log(F_c) + np.log(1 - F_a))

        AD = -size_g - sum_ad / size_g

        return np.sqrt(AD)

    def plot_training_path(self, paths, scen_size):

        plot1 = plt.figure(1)
        for i in range(scen_size):
            plt.title("Geometic Brownian Motion")
            plt.plot(paths[i, :])
            # sns.kdeplot(paths, label='{}th day'.format(str(i)))
            plt.savefig("images/path_gbm_{}100.png".format(str(method)), bbox_inches='tight')
        # plt.show()

    def plot_training_dstr(self, paths, unknown_days):

        plt.title("Geometic Brownian Motion")
        for i in range(unknown_days):
            path = paths[:, i]
            xmin = path.min()
            xmax = path.max()
            x = np.linspace(xmin, xmax, 1000)
            s, loc, scale = stats.lognorm.fit(path, floc=0)
            estimated_mu = np.log(scale)
            estimated_sigma = s
            pdf = stats.lognorm.pdf(x, s, scale=scale)
            plt.plot(x, pdf, 'k')
            plt.show()
            '''mean = np.mean(path)
            var = np.var(path)
            std = np.sqrt(var)
            variance = np.log(1+var/mean**2)

            mu = np.log(mean)-variance/2
            sigma = np.sqrt(variance)

            print(mu)
            print(variance)
            print(sigma)
            #count, bins, ignored = plt.hist(path, 100, normed=True, align='mid')

            #shape, loc, scale = stats.lognorm.fit(path)
            #x=np.linspace(0,6,200)
            dist = stats.lognorm([sigma],loc=mu)

            #pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))/ (x * sigma * np.sqrt(2 * np.pi)))

            x = np.linspace(0,6,10000)
            #pdf = stats.lognorm.pdf(x, shape, loc, scale)


            plt.plot(x, stats.lognorm.pdf(x, mu, sigma))'''
            plt.show()

    def plot_path(self, paths, scen_size,save):


        for i in range(scen_size):
            plt.title("WGAN")
            plt.plot(paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Prediction Days')
        if save:
            plt.savefig("images/path_wgan_{}100.png".format(str(method)), bbox_inches='tight')
        plt.show()

    def plot_2path(self,gbm_paths,model_paths,method):

        fig = plt.figure(figsize=(8, 3))
        #fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05)

        paths = gbm_paths
        for k in range(2):
            if k ==0:
                ax = fig.add_subplot(1, 2, k+1)
            else:
                ax = fig.add_subplot(1, 2, k + 1,sharey=ax)
            #for i in range(1,scen_size):
            for i in range(scen_size):
                ax.plot(paths[i, :])
            if k == 0:
                ax.title.set_text("Geometric Brownian Motion")
            else:
                ax.title.set_text("WGAN-{}".format(str(method)))
            paths = model_paths

        print(len(gbm_paths))
        print(len(model_paths))


        if save:
            fig.savefig("images/path_wgan_{}1000.png".format(str(method)), bbox_inches='tight')
        plt.show()



if __name__ == "__main__":

    # BM
    number_inputs = 100
    unknown_days = 11
    mu = 0
    sigma = 0.1
    method = "brownian"
    iteration = 20000
    save = False

    gbm = Geometric_BM(number_inputs, unknown_days, mu, sigma)
    scen_size = gbm.scen_size
    sigma = gbm.sigma
    s0 = gbm.So
    paths = gbm.predict_path()
    #paths = paths[1:,:]

    graph = Plot_result()


    # GAN-distribution
    wgan = WGAN(paths, number_inputs,method)
    wgan_samples = {}
    o_samples = {}
    paths_pred =wgan.train(iteration)
    #paths_pred = [p * s0 for p in paths_pred]

    for idx in range(unknown_days):
        Simu = Simulation(idx_elment=idx)

        new_samples = Simu.collect_pts(paths_pred)
        wgan_samples[str(idx)] = new_samples

        original_samples = Simu.collect_pts(paths)
        o_samples[str(idx)] = original_samples




    graph.plot_dstr_set(wgan_samples, o_samples, unknown_days,s0, save)

    # Plot GAN-paths
    graph.plot_2path(paths, paths_pred,method)
    plt.show()



