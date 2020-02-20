import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import quandl
import pandas as pd
from stoch_process import Geometric_BM

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

        return np.random.uniform(-1, 1, size=[m, n])
        #np.random.normal(-1, 1, int(self.N))
        #return np.random.normal(0, 1, size=[m, n])


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
        plt.savefig('images/distribution/gan_{}.png'.format(str(idx)))
        #plt.savefig('images/out/gan_{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        plt.show()

        #input()

    def plot_training(self, paths, scen_size):

        plot1 = plt.figure(1)
        for i in range(scen_size):
            plt.title("Geometic Brownian Motion")
            plt.plot(paths[i,:])
            plt.ylabel('SSample values')
            plt.xlabel('Prediction Days')
        plt.show()

    def plot_path(self,paths,scen_size):

        for i in range(scen_size):
            plt.title("GAN")
            plt.plot(paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Prediction Days')
        plt.show()

if __name__ == "__main__":

    #BM
    number_inputs = 1000
    unknown_days = 10
    gbm = Geometric_BM(number_inputs,unknown_days)
    scen_size = gbm.scen_size
    sigma = gbm.sigma
    paths = gbm.predict_path()
    paths = paths[:,1:]


    #GAN-distribution
    gan = GAN(paths, number_inputs)
    gan.train(10000)
    idx = 1
    pathsDstr_pred = gan.predict(paths)
    Simu = Simulation(idx_elment=idx)
    original_samples = Simu.collect_pts(paths)
    new_samples = Simu.collect_pts(pathsDstr_pred)
    # Plot
    graph = Plot_result()
    graph.plot_dstr(original_samples, new_samples,idx)



    # GAN-paths
    '''gan = GAN(paths, number_inputs)
    gan.train(10000)
    paths_pred = gan.predict(paths)
    #Plot
    graph = Plot_result()
    plot_size = scen_size
    graph.plot_training(paths,plot_size)
    graph.plot_path(paths_pred, plot_size)'''



