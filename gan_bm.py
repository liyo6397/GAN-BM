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

class GAN():

    def __init__(self,paths, number_inputs):

        #Input
        self.paths = paths
        #self.layers = layers
        self.N = paths.shape[0]
        self.unknown_days = paths.shape[1]

        # GAN
        self.num_units = 10
        self.paths_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])
        self.z_tf = tf.placeholder(tf.float64, shape=[None, self.paths.shape[1]])

        #G & D
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.G_sample_act, self.G_sample = self.generator(self.z_tf)
        #self.G_sample_tf = tf.placeholder(tf.float32, shape=[None, self.paths.shape[1]])
        self.D_output_real, self.D_logits_real = self.discriminator(self.paths)
        self.D_output_fake, self.D_logits_fake = self.discriminator(self.G_sample)

        # Loss function
        self.D_loss = -tf.reduce_mean(tf.log(self.D_output_real) + tf.log(1. - self.D_output_fake))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_output_fake))
        #self.D_fake = tf.reduce_mean(self.D_output_fake)

        #Optimizing
        self.D_solver, self.G_solver = self.optimizer()

        self.sess.run(tf.global_variables_initializer())

    def sample_Z(self,m,n):

        # random distribution
        #z_process= self.random_distribution(m,n)

        # Brownian Motion
        z_process = self.Brownian()

        return z_process

    def random_distribution(self,m,n):

         #return np.random.uniform(-1, 1, size=[m, n])
         #np.random.normal(-1, 1, int(self.N))
         return np.random.normal(0, 1, size=[m, n])

    def Brownian(self):

        b = {str(scen): np.random.normal(0, 1, int(self.unknown_days)) for scen in range(1, self.N + 1)}
        W = {str(scen): b[str(scen)].cumsum() for scen in range(1, self.N + 1)}

        W_process = np.array([W[str(scen)] for scen in range(1, self.N + 1)])

        return W_process


    def generator(self, z):

        with tf.variable_scope("gen", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs=z, units=self.num_units,
                                      activation=tf.nn.leaky_relu)

            #for l in range(self.layers-2):
            #    hidden = tf.layers.dense(inputs=hidden, units=self.num_units,
            #                             activation=tf.nn.leaky_relu)

            output = tf.layers.dense(inputs=hidden, units=self.paths.shape[1])
            output_log = tf.sigmoid(output)

            return output_log, output

    def discriminator(self, z):

        with tf.variable_scope("disc", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs=z, units=self.num_units,
                                     activation=tf.nn.relu)

            #for l in range(self.layers - 2):
            #hidden = tf.layers.dense(inputs=hidden, units=self.num_units,
            #                             activation=tf.nn.leaky_relu)

            logits = tf.layers.dense(hidden, units=1)
            output = tf.sigmoid(logits)

            return output,logits


    def optimizer(self):

        # Select parameters
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]

        # Optimizers
        D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=disc_vars)
        G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=gen_vars)

        return D_solver, G_solver

    def train(self,itrs):

        tf_dict = {self.paths_tf: self.paths, self.z_tf: self.sample_Z(self.N,self.paths.shape[1])}

        start_time = time.time()
        for it in range(itrs):
            if it % 1000 == 0:
                new_samples = self.sess.run(self.G_sample, tf_dict)
                #print(new_samples)

            # Run disciminator solver
            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss], tf_dict)

            # Run generator solver
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], tf_dict)

        # Print loss
            if it % 1000 == 0:
                #if it % 100 == 0:
                print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))
                #if it == itrs-1:
                #    print("Iteration: %d [D loss: %f] [G loss: %f]" % (it, D_loss_curr, G_loss_curr))

    def predict(self, paths):

        tf_dict = {self.paths_tf: paths, self.z_tf: self.sample_Z(paths.shape[0],paths.shape[1])}

        new_samples = self.sess.run(self.G_sample, tf_dict)

        return new_samples

    def predict_path(self, paths):

        tf_dict = {self.paths_tf: paths, self.z_tf: self.sample_Z(self.N,self.paths.shape[1])}

        new_samples = self.sess.run(self.G_sample, tf_dict)

        return new_samples

class Simulation():

    def __init__(self, idx_elment):

        self.idx_elm = idx_elment-1


    def collect_pts(self,paths):

        sample_dstr = []

        for vec in paths:
            sample_dstr.append(vec[self.idx_elm])

        sample_dstr = np.array(sample_dstr)

        return sample_dstr

    def path_lognormal_dstr(self,samples):

        mean = np.mean(samples)
        std = np.std(samples)
        new_samples = np.random.lognormal(mean,std, len(samples))

        return new_samples

class Plot_result():

    def plot_dstr(self,original_samples,new_samples,idx):
        # Plotting the simulations

        plot1 = plt.figure(1)
        plt.title("Geometric Brownian Motion-{}th day".format(str(idx)))
        plt.hist(original_samples)
        #plt.savefig('images/out/gbm_{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        #plt.show()

        plot2 = plt.figure(2)
        plt.title("Samples generated by GAN-{}th day".format(str(idx)))
        plt.hist(new_samples)
        #plt.savefig('images/distribution/gan_{}.png'.format(str(idx)))
        #plt.savefig('images/out/gan_{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        plt.show()

    def z_value(self,path_gan,path_org):



        mu_g = np.mean(path_gan)

        var_g = np.std(path_gan)/np.sqrt(len(path_gan))

        #s, mu_o, scale = stats.lognorm.fit(path_org, floc=0)
        #var_o = s/np.sqrt(len(path_org))
        mu_o = np.mean(path_org)
        sigma_o = np.std(path_org)/np.sqrt(len(path_org))


        z = (mu_g-mu_o)/(np.sqrt(var_g+sigma_o**2))

        return z

    def normalize(self,data):

        xmin = data.min()
        xmax= data.max()
        data_range = xmax-xmin

        for i, val in enumerate(data):
            data[i] = (val-xmin)/data_range

        return data



    def plot_dstr_set(self,gan_samp, org_samp, n_ipts):

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        paths = np.array([gan_samp[str(scen)] for scen in range(n_ipts)])
        paths_org = np.array([org_samp[str(scen)] for scen in range(n_ipts)])

        #plt.text(0, 0.1, r"The distribution of noise data: Brownian Motion")

        for scen in range(n_ipts):
            plt.title("Data distribution:\n Geometric Brownian Motion and generator sampling-{}th day".format(str(scen)))
            path = paths[scen,:]
            path_org = paths_org[scen,:]



            xmin = min(path_org.min(),path.min())
            xmax = max(path_org.max(),path.max())

            s, mu, scale = stats.lognorm.fit(path_org, floc=0)
            x = np.linspace(0, xmax, 10000)

            pdf = stats.lognorm.pdf(x, s, loc=mu, scale=scale)
            #norm_path = self.normalize(path)
            #norm_path_org = self.normalize(path_org)
            #A, critical, sig = stats.anderson_ksamp([path,pdf])
            #print(A)
            #print(sig)

            #sns.kdeplot(pdf)
            plt.plot(x, pdf, 'k', label="Geometric Brownian Motion")
            #sns.kdeplot(path)

            plt.hist(path,50,density=True,stacked=True)
            #plt.text(0, 0.1, r'$AD value: {}$'.format(str(A)))
            #plt.legend()
            plt.savefig("dstr{}_bm_gan.png".format(str(scen)), bbox_inches='tight')
            plt.show()

    def aderson_test(self,g_sample,s,mu,scale):

        size_g = len(g_sample)
        g_sample.sort()
        sum_ad = 0

        for i in range(size_g):
            F_c = stats.lognorm(s=s, loc=mu,scale=scale).cdf(g_sample[i])
            F_a = stats.lognorm(s=s, loc=mu,scale=scale).cdf(g_sample[size_g-i-1])

            sum_ad += (2*(i+1)-1)*(np.log(F_c)+np.log(1-F_a))

        AD = -size_g-sum_ad/size_g

        return np.sqrt(AD)









    def plot_training_path(self, paths, scen_size):

        plot1 = plt.figure(1)
        for i in range(scen_size):
            plt.title("Geometic Brownian Motion")
            plt.plot(paths[i,:])
            #sns.kdeplot(paths, label='{}th day'.format(str(i)))
        #plt.show()


    def plot_training_dstr(self, paths, unknown_days):

        plt.title("Geometic Brownian Motion")
        for i in range(unknown_days):
            path = paths[:,i]
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

    def plot_path(self,paths,scen_size):

        for i in range(scen_size):
            plt.title("GAN")
            plt.plot(paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Prediction Days')
        #plt.show()

if __name__ == "__main__":

    #BM
    number_inputs = 2000
    unknown_days = 10
    mu = 0
    sigma = 0.1





    gbm = Geometric_BM(number_inputs,unknown_days,mu,sigma)
    scen_size = gbm.scen_size
    sigma = gbm.sigma
    paths = gbm.predict_path()
    paths = paths[:,1:]
    graph = Plot_result()
    #graph.plot_training_dstr(paths, unknown_days)





    #GAN-distribution
    gan = GAN(paths, number_inputs)
    gan_samples = {}
    o_samples = {}
    gan.train(20000)
    pathsDstr_pred = gan.predict(paths)
    
    for idx in range(unknown_days):
        Simu = Simulation(idx_elment=idx)

        new_samples = Simu.collect_pts(pathsDstr_pred)
        gan_samples[str(idx)] = new_samples

        original_samples = Simu.collect_pts(paths)
        o_samples[str(idx)] = original_samples

    # Plot


    graph.plot_dstr_set(gan_samples, o_samples, unknown_days)


    # GAN-paths
    '''gan = GAN(paths, number_inputs)
    gan.train(20000)
    plot_size = 1000
    paths_pred = gan.predict(paths)
    #Plot
    #plot_size = scen_size
    graph.plot_training_path(paths,plot_size)
    plt.show()
    graph.plot_path(paths_pred, plot_size)
    plt.show()'''



