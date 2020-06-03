import numpy as np

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