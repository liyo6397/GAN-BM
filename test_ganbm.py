import unittest
from gan_bm import GAN, Simulation
from stoch_process import Geometric_BM
import tensorflow as tf


class Test_GAN(unittest.TestCase):

    def setUp(self):
        self.BM = Geometric_BM(number_inputs=5,time_window=3)

        self.paths = self.BM.predict_path()
        self.layers = 3
        self.num_units = 10
        self.time_window = self.paths.shape[1]
        self.gan = GAN(self.paths, self.layers)
        self.sess = self.gan.sess
        #self.sess.run(tf.global_variables_initializer())




    def test_BMvariable(self):

        data = self.paths
        print(data)
        time_window = data.shape[1]
        print(time_window)

    def test_tfSession(self):
        self.gan = GAN(self.paths, self.layers)

        paths_tf = self.gan.sess.run([self.gan.paths_tf], feed_dict={self.gan.paths_tf: self.paths})


        print(paths_tf)

    def test_Zsample(self):

        #self.gan = GAN(self.paths, self.layers)

        z_tf = self.gan.sess.run([self.gan.z_tf], feed_dict={self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1])})

        print(z_tf)

    def test_generator(self):

        G_sample = self.sess.run([self.gan.G_sample], feed_dict={self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1])})

        print(G_sample)

    def test_discriminator_outputReal(self):

        D_output_real, D_logits_real = self.gan.sess.run([self.gan.D_output_real, self.gan.D_logits_real], feed_dict={self.gan.paths_tf: self.paths})

        print(D_output_real)
        print(D_logits_real)

    def test_discriminator_outputFake(self):
        feed_dict = {self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1])}



        #G_sample_tf = tf.convert_to_tensor(g_sample)

        D_output_fake, D_logits_fake = self.sess.run([self.gan.D_output_fake, self.gan.D_logits_fake],feed_dict)

        print(D_output_fake)
        print(D_logits_fake)

    def test_loss(self):
        feed_dict = {self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1]), self.gan.paths_tf: self.paths}

        print("D loss")
        D_loss = self.sess.run([self.gan.D_loss],feed_dict)
        print(D_loss)
        print("G loss")
        G_loss = self.sess.run([self.gan.G_loss], feed_dict)
        print(G_loss)

    def test_optimization(self):
        feed_dict = {self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1]), self.gan.paths_tf: self.paths}

        #D_loss = self.sess.run([self.gan.D_loss],feed_dict)

        D_solver, D_loss_curr = self.sess.run([self.gan.D_solver, self.gan.D_loss], feed_dict)
        G_solver, G_loss_curr = self.sess.run([self.gan.G_solver, self.gan.G_loss], feed_dict)

        print(D_loss_curr)
        print(G_loss_curr)
        #print(D_loss)

    def test_predict(self):
        feed_dict = {self.gan.z_tf: self.gan.sample_Z(5, self.paths.shape[1]), self.gan.paths_tf: self.paths}

        new_samples = self.sess.run(self.gan.G_sample, feed_dict)

        print(new_samples)

    def test_sample(self):

        samples_path = self.gan.predict(self.paths)

        simulation = Simulation(idx_elment=2)

        new_samples = simulation.collect_pts(samples_path)

        print(samples_path)
        print(samples_path[:,1:])
        print(new_samples)





















