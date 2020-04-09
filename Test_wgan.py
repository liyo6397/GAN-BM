import unittest
from wgan_bm import WGAN, Simulation, Plot_result
from stoch_process import Geometric_BM
import tensorflow as tf
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.number_inputs = 100
        self.unknown_days = 10
        self.mu = 0
        self.sigma = 0.1
        self.method = "uniform"
        self.converge_crit = 10**-5
        self.save = False

        self.batch_size = 10
        self.print_itr = 1000

        self.gbm = Geometric_BM(self.number_inputs, self.unknown_days, self.mu, self.sigma)
        self.graph = Plot_result(self.save)
        self.paths = self.gbm.predict_path()

        self.wgan = WGAN(self.paths, self.method,self.batch_size)
        self.sess = self.wgan.sess


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


        self.graph.plot_dstr_set(g_sam, r_sam, self.unknown_days, self.gbm.So,self.mu, self.sigma)
        self.graph.plot_2path(self.paths, pred_paths, self.method)

    def test_train_loss(self):

        pred_paths, loss_d, loss_g = self.wgan.train(self.converge_crit,self.print_itr)

        Sim = Simulation(self.unknown_days, self.paths, pred_paths)

        r_sam, g_sam = Sim.samples()

        self.graph.loss_plot(loss_d,loss_g)
        self.graph.plot_dstr_set(g_sam, r_sam, self.unknown_days, self.gbm.s0, self.mu, self.sigma)
        self.graph.plot_dstr_set_hist(g_sam, r_sam, self.unknown_days, self.gbm.s0, self.mu, self.sigma)
        self.graph.plot_2path(self.paths, pred_paths, self.method, self.gbm.s0)

    def test_Gloss(self):

        loss_g = self.sess.run([self.wgan.G_loss], feed_dict={self.wgan.z_tf: self.wgan.sample_Z(self.number_inputs, self.paths.shape[1])})

        print(loss_g)




