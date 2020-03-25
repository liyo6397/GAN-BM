import numpy as np
class Stats_test():

    def z_value(self, path_gan, path_org):

        mu_g = np.mean(path_gan)

        var_g = np.std(path_gan) / np.sqrt(len(path_gan))

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

    def norm2(self, distance):

        dis_sum = np.zeros(self.N)

        for i, dis in enumerate(distance):
            dis_sum[i] = np.sum(dis)

        max_sum = np.max(np.abs(dis_sum))

        return max_sum

    def lip_loss(self):

        # self.paths_tf = self.sess.run(self.paths_tf,feed_dict={self.paths_tf: paths})

        v = 0.01

        distance = self.paths - self.g_samp
        norm2_dis = np.linalg.norm(distance, 2)

        # norm2_dis = self.norm2(distance)

        lip = np.abs(np.abs(self.D_output_real - self.D_output_fake) / tf.abs(norm2_dis) - 1.)

        lip_sum = tf.reduce_sum(lip)

        lip = v * lip_sum

        # D_loss = -tf.reduce_mean(self.D_output_real) + tf.reduce_mean(self.D_output_fake)+lip
        # G_loss = -tf.reduce_mean(self.D_output_fake)

        return lip

