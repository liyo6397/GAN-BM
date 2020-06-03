import numpy as np
from wgan_bm import WGAN
from plot_result import plot_result
from data_collect import Simulation
from gan_bm import GAN
from finite_dimensional import joint_distribution
from stoch_process import Geometric_BM, Orn_Uh



class problem_setup():

    def __init__(self,method):
        self.number_inputs = 100
        self.converge_crit = 10 ** (-6)
        self.print_itr = 1000
        self.itrs = 100000
        self.method = method


    def Brownian_motion(self):

        unknown_days = 10
        mu = 0
        sigma = 0.1
        data_type = "Geometric Brownian Motion"
        s0 = 1

        gbm = Geometric_BM(self.number_inputs, unknown_days, mu, sigma, s0)
        paths = gbm.predict_path()

        return mu, sigma, data_type, s0, paths

    def Orn_Uh(self):

        unknown_days = 10
        mu = 0.8
        sigma = 0.3
        theta = 1.1
        s0 = 0.01
        t0 = 0
        tend = 2

        data_type = "Ornstein-Uhlenbeck process"

        orn = Orn_Uh(self.number_inputs, unknown_days, mu, sigma, theta, s0, t0, tend)
        paths = orn.predict_path()

        return mu, sigma, theta, t0, tend, data_type, s0, paths

if __name__ == "__main__":



    save = False
    process = "ou"
    model = "wgan"
    method = "brownian"

    setup = problem_setup(method)

    # Geometric Brownian Motion
    if process == "gbm":
        mu, sigma, data_type, s0, paths = setup.Brownian_motion()

    #Ornstein-Uhlenbeck
    if process == "ou":
        mu, sigma,theta, t0, tend, data_type, s0, paths = setup.Orn_Uh()

    # GAN
    if model == "gan":
        gan = GAN(paths[:, 1:], method)
        paths_pred, loss_d, loss_g = gan.train(setup.itrs, setup.print_itr)

    # WGAN
    if model == "wgan":
        wgan = WGAN(paths[:,1:], method)
        paths_pred, loss_d, loss_g =wgan.train(setup.converge_crit, setup.print_itr)

    Nx = 50
    Ny = 50
    fd = joint_distribution(Nx, Ny, mu, sigma, s0, paths_pred)

    X, Y, p_T, p_data = fd.grid_loss()

    loss = p_T - p_data
    loss = np.abs(loss)
    print(np.mean(loss))

    # Data Collection
    unknown_days = paths.shape[1]-1
    Sim = Simulation(unknown_days, paths, paths_pred)
    r_samples, g_samples = Sim.samples()

    # Plot
    graph = plot_result(save, data_type, method)
    graph.loss_plot(loss_d,loss_g)
    graph.plot_2path(paths, paths_pred, method, s0)

    if process == "gbm":
        graph.dstr_gbm(g_samples, unknown_days, s0, mu, sigma)
    if process == "ou":
        graph.dstr_ou(g_samples, unknown_days, s0, sigma, theta, t0, tend)
    graph.fdd_plot(X, Y, loss)

    print(loss.shape)





