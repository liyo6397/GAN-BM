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

    def test_tangent(self):

        gbm = Geometric_BM(number_inputs=10, time_window=5, mu=0, sigma=1)
        paths = gbm.predict_path()

        Nx = 10
        Ny = 10
        fd_distribution = joint_distribution(paths, Nx, Ny)

        tan = fd_distribution.tangent_timezone()

        print(tan)

    def test_y_grid_value(self):

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=0, sigma=0.1, s0=0.01)
        paths = gbm.predict_path()


        Nx = 20
        Ny = 20
        fd_distribution = joint_distribution(paths, Nx, Ny, 0, 1)


        time_steps = 100
        T_to_s, time_st = fd_distribution.y_grid_values(time_steps)
        time = np.linspace(0, fd_distribution.T, time_steps + 1)

        t_plot = np.linspace(0,fd_distribution.T,paths.shape[1])

        print(T_to_s)

        for i in range(paths.shape[0]):
            plt.title("WGAN Sampling")
            plt.plot(t_plot, paths[i, :])
            plt.ylabel('Sample values')
            plt.xlabel('Time')

        for i in range(T_to_s.shape[1]):
            plt.plot(time, T_to_s[:, i], '^')

        plt.show()


    def test_fd_theoritical(self):

        mu = 0
        sigma = 0.1
        s0 = 0.01

        gbm = Geometric_BM(number_inputs=100, time_window=10, mu=mu, sigma=sigma, s0=s0)
        paths = gbm.predict_path()

        Nx = 20
        Ny = 20
        fd= joint_distribution(paths, Nx, Ny, mu, sigma)


        start = 0.008
        end = 0.009
        s = np.linspace(start, end, 10)
        delta_t = paths.shape[1]/Ny

        time = np.linspace(0,fd.T,paths.shape[1])

        pdf = fd.fd_theoritical(s,delta_t)

        print(pdf)

    def test_num_pts_grid(self):

        mu = 0
        sigma = 0.1

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=mu, sigma=sigma, s0=1)
        paths = gbm.predict_path()

        Nx = 20
        Ny = 20
        fd = joint_distribution(paths, Nx, Ny, mu, sigma)

        num_steps = 1000
        grid_st, time_st = fd.y_grid_values(num_steps)

        num_pts = fd.num_pts_grid(grid_st, time_st)

        print(num_pts)

    def test_fd_match(self):

        mu = 0
        sigma = 0.1
        s0 =0.01

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=mu, sigma=sigma, s0=s0)
        paths = gbm.predict_path()

        Nx = 30
        Ny = 30
        fd = joint_distribution(paths, Nx, Ny, mu, sigma)

        y = fd.y

        grids_diff = (y[-1] - y[0]) / Nx

        num_steps = 100
        grid_st, time_st = fd.y_grid_values(num_steps)
        grids_density = fd.num_pts_grid(grid_st, time_st)

        start = 5
        end = 10
        grid_density = fd.grid_density(grids_density, start, end, num_steps)

        num_grids = (y[end] - y[start]) / grids_diff
        print("start", y[start])
        print("end", y[end])

        s = np.linspace(fd.y[start],fd.y[end], 10)
        #s = paths[1,:]
        delta_t = 1

        # print("num_grids", num_grids)

        #theo_pdf = fd.fd_theoritical_3(s, delta_t, s0)
        #theo_pdf = fd.cov_multi(s, s0)
        theo_pdf = fd.BM_joint_dstr(s, s0, paths[1,:], delta_t)

        print("Theoritical: ",theo_pdf)
        print("WGAN: ",grid_density)

    def test_cov(self):

        mu = 0
        sigma = 0.1

        gbm = Geometric_BM(number_inputs=10, time_window=10, mu=mu, sigma=sigma, s0=0.01)
        paths = gbm.predict_path()

        Nx = 30
        Ny = 30

        fd = joint_distribution(paths, Nx, Ny, mu, sigma)


        s = np.linspace(fd.y[5],fd.y[6],10)

        cov = fd.cov_multi(s,0.01)

        print(cov)

    def test_match_central_limit(self):

        mu = 0
        sigma = 0.1
        s0 = 0.01
        num_time = 10

        gbm = Geometric_BM(number_inputs=10, time_window=num_time, mu=mu, sigma=sigma, s0=s0)
        paths = gbm.predict_path()

        method = "uniform"
        converge_crit = 10**(-4)
        wgan = WGAN(paths[:,1:], method)
        print_itr = 1000
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)

        Nx = 30
        Ny = 30

        fd = joint_distribution(paths_pred, Nx, Ny, mu, sigma)
        t_range = 10
        target1 = fd.y[15]
        print(target1)
        p_value1 = fd.central_limit(target1,s0, 1)
        print(p_value1)
        target2 = fd.y[20]
        print(target2)
        p_value2 = fd.central_limit(target2, s0, 5)
        print(p_value2)


        ans = p_value2-p_value1

        num_steps = 100
        data, time_st = fd.y_grid_values(num_steps)

        time_range = int(5/(num_time/100))

        count = 0


        for i in range(time_range):
            for j in range(data.shape[1]):
                if data[i,j] >= target1 and data[i,j] <= target2:
                    count += 1

        pdf_data = count/(time_range*data.shape[1])

        print(pdf_data)


        print(ans)

    def test_central_limit(self):

        mu = 0
        sigma = 0.1
        s0 = 0.01
        num_time = 10

        gbm = Geometric_BM(number_inputs=10, time_window=num_time, mu=mu, sigma=sigma, s0=s0)
        paths = gbm.predict_path()

        Nx = 30
        Ny = 30

        fd = joint_distribution(paths, Nx, Ny, mu, sigma,s0)


        target1 = fd.y[0]
        target2 = fd.y[29]
        print("target1: ", target1)
        print("target2: ", target2)

        #p1 = fd.central_limit(0.1274,s0,1)
        #p2 = fd.central_limit(0.29997, s0, 5)
        p1 = fd.central_limit(target1,s0,1)
        p2 = fd.central_limit(target2, s0, 5)

        print(p1)
        print(p2)

    def test_fdd_plot(self):

        mu = 0
        sigma = 0.1
        s0 = 0.01
        num_time = 10

        gbm = Geometric_BM(number_inputs=10, time_window=num_time, mu=mu, sigma=sigma, s0=s0)
        paths = gbm.predict_path()

        Nx = 20
        Ny = 20



        method = "uniform"
        wgan = WGAN(paths[:, 1:], method)

        converge_crit = 10 ** (-4)
        print_itr = 100
        paths_pred, loss_d, loss_g = wgan.train(converge_crit, print_itr)

        fd = joint_distribution(Nx, Ny, mu, sigma, s0, paths_pred)
        X, Y, p_T, p_data = fd.grid_loss()


        loss = p_T-p_data



        print("loss: ", loss)
        print(Y)

        #plt(loss)
        #plt.colorbar()
        #plt.show()






























