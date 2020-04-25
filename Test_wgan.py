import unittest
from wgan_bm import WGAN, Simulation, Plot_result
from stoch_process import Geometric_BM, Orn_Uh
from finite_dimensional import joint_distribution
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class Test_Wgan(unittest.TestCase):
    def setUp(self):
        self.number_inputs = 100
        self.unknown_days = 10
        self.mu = 0
        self.sigma = 0.1
        self.method = "uniform"
        self.converge_crit = 10**-2
        self.save = False

        self.output_units = 20
        self.print_itr = 1000

        self.gbm = Geometric_BM(self.number_inputs, self.unknown_days, self.mu, self.sigma)

        self.paths = self.gbm.predict_path()

        #self.wgan = WGAN(self.paths, self.method,self.output_units)
        #self.sess = self.wgan.sess


    def test_train(self):
        feed_dict = {self.wgan.z_tf: self.wgan.sample_Z(self.number_inputs, self.paths.shape[1]), self.wgan.paths_tf: self.paths}

        pred_paths = self.wgan.train(self.iteration)

        Sim = Simulation(self.unknown_days, self.paths, pred_paths)

        r_sam, g_sam = Sim.samples()

        self.graph.plot_dstr_set(g_sam, r_sam, self.unknown_days, self.gbm.So)
        self.graph.plot_2path(self.paths, pred_paths, self.method)

    def test_nbatch(self):

        n_batch = self.wgan.create_mini_batches(self.wgan.paths,self.batch_size)


        print(np.shape(n_batch))
        print(np.shape(n_batch[1]))
        print(np.shape(self.wgan.paths))


    def test_batchVar(self):
        n_batch = self.wgan.create_mini_batches(self.paths,self.batch_size)


        paths_tf = self.sess.run([self.wgan.paths_tf], feed_dict={self.wgan.paths_tf: n_batch[0]})
        print(paths_tf)
        z_tf = self.sess.run([self.wgan.z_tf],feed_dict={self.wgan.z_tf: self.wgan.sample_Z(n_batch[0].shape[0], n_batch[0].shape[1])})
        print(z_tf)

    def test_nbatchloss(self):
        paths_batch = self.wgan.create_mini_batches(self.paths, self.batch_size)
        print(np.shape(paths_batch))
        print(np.shape(self.paths))

        count = 0

        for b,batch in enumerate(paths_batch):
            tf_dict = {self.wgan.paths_tf: batch,
                       self.wgan.z_tf: self.wgan.sample_Z(batch.shape[0], batch.shape[1])}

            D_loss = self.sess.run([self.wgan.D_loss], tf_dict)
            print("Number of batch: %d" % (b))
            print(D_loss)

        print(count)



    def test_trainnbatch(self):
        print("Number of real samples:")
        print(np.shape(self.paths))

        pred_paths = self.wgan.train_nbatch2(self.iteration,self.print_itr)
        print("Number of g samples:")
        print(np.shape(pred_paths))

        Sim = Simulation(self.unknown_days, self.paths, pred_paths)

        r_sam, g_sam = Sim.samples()


        self.graph.plot_dstr_set(g_sam, r_sam, self.unknown_days, self.gbm.s0,self.mu, self.sigma)
        self.graph.plot_2path(self.paths, pred_paths, self.method, self.gbm.s0)

    def test_train_loss(self):

        pred_paths, loss_d, loss_g = self.wgan.train(self.converge_crit,self.print_itr)

        Sim = Simulation(self.unknown_days, self.paths, pred_paths)

        r_sam, g_sam = Sim.samples()

        #self.graph.loss_plot(loss_d,loss_g)
        self.graph.plot_dstr_set(g_sam, r_sam, self.unknown_days, self.gbm.s0, self.mu, self.sigma)
        self.graph.plot_dstr_set_hist(g_sam, r_sam, self.unknown_days, self.gbm.s0, self.mu, self.sigma)
        self.graph.plot_2path(self.paths, pred_paths, self.method, self.gbm.s0)

    def test_Gloss(self):

        loss_g = self.sess.run([self.wgan.G_loss], feed_dict={self.wgan.z_tf: self.wgan.sample_Z(self.number_inputs, self.paths.shape[1])})

        print(loss_g)


    def test_output_pts(self):

        #print(np.shape(self.gbm.predict_path()))

        pred_paths, loss_d, loss_g = self.wgan.train(self.converge_crit, self.print_itr)

        final_sample = self.wgan.generator(self.wgan.z_tf, output_units = 20)

        self.g_samp = self.sess.run(final_sample, feed_dict={self.wgan.z_tf: self.wgan.sample_Z(100, self.paths.shape[1])})

        self.graph.plot_2path(self.paths, pred_paths, self.method, self.gbm.s0)




    def test_orn(self):
        number_inputs = 10
        unknown_days = 30
        mu = 0.8
        sigma = 0.3
        theta = 1.1
        s0 = 0
        t0 = 0
        tend = 2

        method = "uniform"
        converge_crit = 10 ** (-4)
        print_itr = 1000
        save = False
        data_type = "Ornstein-Uhlenbeck process"

        orn = Orn_Uh(number_inputs, unknown_days, mu, sigma, theta, s0, t0, tend)

        paths = orn.predict_path()
        wgan = WGAN(paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)

        graph = Plot_result(save, data_type)

        graph.plot_2path(paths, paths_pred, method, s0)

class Test_joint_distribution(unittest.TestCase):


    def test_coordinates(self):

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=0, sigma=1)
        paths = gbm.predict_path()

        Nx = 5
        Ny = 5
        fd_distribution = joint_distribution(paths,Nx, Ny)

        x, y = fd_distribution.setup_coord_value()

        print("T: ", fd_distribution.T)
        print("x :", x)
        print("y :", y)

    def test_tangent(self):

        gbm = Geometric_BM(number_inputs=10, time_window=5, mu=0, sigma=1)
        paths = gbm.predict_path()

        Nx = 10
        Ny = 10
        fd_distribution = joint_distribution(paths, Nx, Ny)

        tan = fd_distribution.tangent_timezone()

        print(tan)

    def test_y_grid_value(self):

        gbm = Geometric_BM(number_inputs=5, time_window=5, mu=0, sigma=1)
        paths = gbm.predict_path()

        Nx = 10
        Ny = 10
        fd_distribution = joint_distribution(paths, Nx, Ny)

        T_to_s = fd_distribution.y_grid_values_new()
        plot_tTos = T_to_s.T

        for i in range(paths.shape[0]):
            plt.title("Geometric Brownian Motion")
            plt.plot(paths[i, :])
            plt.plot(plot_tTos[i, :], 'r^')
            plt.ylabel('Sample values')
            plt.xlabel('Time')


        plt.show()


        print(T_to_s[1,:])




