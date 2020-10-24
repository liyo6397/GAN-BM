import unittest
from wgan_bm import WGAN
from stoch_process import Geometric_BM, Orn_Uh
from finite_dimensional import joint_distribution, BM_joint
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_collect import Simulation
from plot_result import plot_result
from MCInt import MC_fdd, dim_reduction



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

        #self.gbm = Geometric_BM(self.number_inputs, self.unknown_days, self.mu, self.sigma)

        #self.paths = self.gbm.predict_path()

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
        unknown_days = 10
        mu = 0.8
        sigma = 0.3
        theta = 1.1
        s0 = 0.01
        t0 = 0
        tend = 2

        method = "uniform"
        converge_crit = 10 ** (-2)
        print_itr = 1000
        save = False
        data_type = "Ornstein-Uhlenbeck process"

        orn = Orn_Uh(number_inputs, unknown_days, mu, sigma, theta, s0, t0, tend)

        paths = orn.predict_path()
        wgan = WGAN(paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)

        graph = Plot_result(save, data_type)
        Sim = Simulation(unknown_days, paths, paths_pred)


        r_samples, g_samples = Sim.samples()
        graph.plot_dstr_set(g_samples, r_samples, unknown_days, s0, mu, sigma)
        graph.plot_2path(paths, paths_pred, method, s0)

class Test_joint_distribution(unittest.TestCase):

    def setUp(self):
        self.num_pts = 5
        self.num_dim = 10
        self.gbm = Geometric_BM(number_inputs=self.num_pts, time_window=self.num_dim, mu=0, sigma=1, s0=1)
        self.paths = self.gbm.predict_path()
        self.mu = 0
        self.sigma = 0.1
        self.s0 = 0.1




    def test_coordinates(self):

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=0, sigma=1, s0=1)
        paths = gbm.predict_path()

        Nx = 5
        Ny = 5
        fd_distribution = joint_distribution(paths,Nx, Ny)

        x, y = fd_distribution.setup_coord_value()

        print("T: ", fd_distribution.T)
        print("time:", fd_distribution.time)
        print("x :", x)
        print("y :", y)

    def test_BM_joint(self):

        nx = 5
        ny = 5
        s0 = 0.1
        fd_distribution = joint_distribution(nx, ny, self.mu, self.sigma, s0)

        print(min(self.paths[:, -1]))
        print(max(self.paths[:,-1]))
        path = np.linspace(0.1,0.2,10)
        delta_t = 0.1
        pdf = fd_distribution.BM_joint_dstr(path, delta_t)

        print(pdf)

    def test_cubic_dens(self):

        nx = 5
        ny = 5

        fd_distribution = joint_distribution(nx, ny, self.mu, self.sigma, self.s0, self.paths)

        fun_pdf = fd_distribution.cubic_dens(0.01, 0.5, 100)
        ct = 0
        for i in range(self.paths.shape[0]):
            for j in range(self.paths.shape[1]):
                if self.paths[i,j] >= 1 and self.paths[i,j] <= 2:
                    ct += 1

        test_pdf = ct / (self.paths.shape[0]*self.paths.shape[1])

        print("test:",test_pdf)
        print(self.paths)
        print("fun:: ", fun_pdf)

    def test_cubid_prob(self):

        nx = 5
        ny = 5

        fd_distribution = joint_distribution(nx, ny, self.mu, self.sigma, self.s0, self.paths)

        probs = fd_distribution.cubid_probability()

        print(probs)

    def test_y_grid_values(self):

        nx = 5
        ny = 5
        time_steps = 100

        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        s_t, time_t = fd.y_grid_values(time_steps, self.paths)

        time = np.linspace(0, self.num_dim, time_steps + 1)

        t_plot = np.linspace(0, self.num_dim, self.paths.shape[1])

        for i in range(self.paths.shape[0]):
            plt.title("WGAN Sampling")
            plt.plot(t_plot, self.paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Time')

        for i in range(s_t.shape[0]):
            plt.plot(time, s_t[i, :], '^')

        plt.show()

    def test_python_JDfunction(self):

        nx = 5
        ny = 5

        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        pdf = fd.BM_joint_dstr2(self.paths)

        print(self.paths)

        print(pdf)

    def test_compaire_2gridLoss(self):

        nx = 5
        ny = 5
        time_steps = 100

        #theoritical
        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        s_t, time_st = fd.y_grid_values(time_steps, self.paths)

        #model
        method = "uniform"
        converge_crit = 10 ** (-2)
        print_itr = 1000
        wgan = WGAN(self.paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)
        s_t_pred, time_t_pred = fd.y_grid_values(time_steps, paths_pred)

        loss_grid = fd.grid_loss(s_t, s_t_pred, time_st)

        print(loss_grid)

    def test_fddplot(self):

        nx = 5
        ny = 5
        time_steps = 100

        # theoritical
        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        s_t, time_st = fd.y_grid_values(time_steps, self.paths)

        # model
        method = "uniform"
        converge_crit = 10 ** (-2)
        print_itr = 1000
        wgan = WGAN(self.paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)
        s_t_pred, time_t_pred = fd.y_grid_values(time_steps, paths_pred)

        den_1, den_2, loss = fd.grid_loss_scatter(s_t, s_t_pred, time_st)

        den_theo, den_emp = fd.fdd_den_by_value(den_1, den_2)

        print("den_theo: ", den_theo)
        print("den_emp: ", den_emp)

        save = "False"
        data_type = "Geometric Brownian Motion"
        method = "uniform"
        graph = plot_result(save, data_type, method)
        #graph.fdd_plot_scatter(self.paths, den_1, den_2, time_st, loss)
        graph.fdd_plot_2Dscatter(self.paths, den_theo, den_emp)

    def test_fdd3D(self):

        nx = 5
        ny = 5
        time_steps = 10

        # theoritical
        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        s_t, time_st = fd.y_grid_values(time_steps, self.paths)

        # model
        method = "uniform"
        converge_crit = 10 ** (-2)
        print_itr = 100
        wgan = WGAN(self.paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)
        s_t_pred, time_t_pred = fd.y_grid_values(time_steps, paths_pred)

        den_1, den_2, loss = fd.meshgrid_loss(s_t, s_t_pred, time_st)

        #print(time_st)
        print(s_t)
        print(s_t_pred)
        print(den_1)
        print(den_2)

        t_plot = np.linspace(0, self.num_dim, self.paths.shape[1])
        time = np.linspace(0, self.num_dim, time_steps + 1)

        '''for i in range(self.paths.shape[0]):
            plt.title("WGAN Sampling")
            plt.plot(t_plot, self.paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Time')

        for i in range(s_t.shape[0]):
            plt.plot(time, s_t[i, :], '^')

        plt.show()'''

        save = "False"
        data_type = "Geometric Brownian Motion"
        method = "uniform"
        graph = plot_result(save, data_type, method)
        graph.fdd_plot_3D(time_st,s_t, s_t_pred, den_1, den_2, loss)

        diff = fd.Relative_Percent_Difference(den_1, den_2)

        print("MAPE: ", diff)

    def test_TheoEmp_loss(self):

        nx = 10
        ny = 10
        time_steps = 100

        # theoritical
        fd = joint_distribution(nx, ny, self.mu, self.sigma, self.s0)
        s_t, time_st = fd.y_grid_values(time_steps, self.paths)

        # model
        method = "uniform"
        converge_crit = 10 ** (-2)
        print_itr = 100
        wgan = WGAN(self.paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)
        s_t_pred, time_t_pred = fd.y_grid_values(time_steps, paths_pred)


        den_1, den_2, loss = fd.TheoEmp_loss(s_t, s_t_pred, time_st)

        save = "False"
        data_type = "Geometric Brownian Motion"
        method = "uniform"
        graph = plot_result(save, data_type, method)
        graph.fdd_plot_3D(time_st, s_t, s_t_pred, den_1, den_2, loss)

        print(den_1)

        diff = fd.Relative_Percent_Difference(den_1, den_2)

        print("RPD: ", diff)

class Test_BM_joint(unittest.TestCase):

    def test_cubid_vector(self):

        sigma = 0.1
        s0 = 0.1
        bm = BM_joint(sigma, s0)

        x_range = np.linspace(0,1,10)

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = bm.cubid_vector(x_range)

        print(x10)

    def test_BM_joint_fun(self):

        sigma = 0.1
        s0 = 0.1
        bm = BM_joint(sigma, s0)
        dim = 10

        x_range = np.linspace(0.1, 0.8, dim)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = bm.cubid_vector(x_range)

        density = bm.BM_joint_fun(dim, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        print(density)

    def test_cubid_prob(self):
        sigma = 0.1
        s0 = 0.1
        bm = BM_joint(sigma, s0)
        dim = 10

        x_range = np.linspace(0.1, 0.8, dim)
        int_val = bm.cubid_prob(x_range)

        print(int_val)

    def test_dataStandardize(self):

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=0, sigma=1, s0=1)
        paths = gbm.predict_path()

        data_scale = dim_reduction()

        stand_data = data_scale.data_standardize(paths)

        features = np.zeros(stand_data.shape[0])

        for i,data in enumerate(stand_data):
            features[i] = np.sum(data)

        print(features)

    def test_principle_component(self):
        gbm = Geometric_BM(number_inputs=1000, time_window=10, mu=0, sigma=1, s0=1)
        paths = gbm.predict_path()

        data_scale = dim_reduction()

        pca_set, pca_pdf = data_scale.principle_component(paths)

        print(pca_set.shape)

    def test_visual(self):
        gbm = Geometric_BM(number_inputs=1000, time_window=10, mu=0, sigma=1, s0=1)
        paths = gbm.predict_path()

        data_scale = dim_reduction()

        pca_components, pca_df = data_scale.principle_component(paths)

        print(pca_components.shape)

        data_scale.visualize(pca_components, pca_df)

    def test_pca_wgan(self):

        gbm = Geometric_BM(number_inputs=100, time_window=10, mu=0, sigma=1, s0=1)
        paths = gbm.predict_path()


        method = "uniform"
        converge_crit = 10**(-2)
        print_itr = 100

        wgan = WGAN(paths[:, 1:], method)
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)

        data_scale = dim_reduction()

        pca_components_theo, pca_df_theo = data_scale.principle_component(paths[:, 1:])
        pca_components_wgan, pca_df_wgan = data_scale.principle_component(paths_pred)

        print(paths.shape)
        print(paths_pred.shape)

        data_scale.visualize(pca_components_theo, pca_df_theo)
        data_scale.visualize(pca_components_wgan, pca_df_wgan)

    def test_Mcint(self):

        sigma = 0.1
        x0 = 0.1
        x_range = [0.1, 0.8]

        MC = MC_fdd(sigma, x0, x_range)
        MC.MC_int()

class Test_MC_fdd(unittest.TestCase):

    def setUp(self):
        self.sigma = 0.1
        self.s0 = 0.1
        self.gbm = Geometric_BM(number_inputs=5, time_window=10, mu=0, sigma=self.sigma, s0=self.s0)
        self.data = self.gbm.predict_path()

        self.MC = MC_fdd(self.sigma, self.s0, self.data)


    def test_multi_mean(self):

        mu = self.MC.multi_mean(self.data)

        print(mu)

    def test_standardized(self):

        drift = self.gbm.drift
        W = self.MC.standardized(self.data[1:,1:], drift)

        print(W)

    def test_extract_bm(self):
        drift = self.gbm.drift
        self.data = self.data[1:,1:]
        bm, stand_bm = self.MC.extract_bm(self.data, drift)

        print("Brownian Motion: ")
        print(bm)
        print("Standardized Brownian Motion: ")
        print(stand_bm)










































