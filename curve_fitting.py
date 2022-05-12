import numpy as np

class curve_fit_decay_func():
    @staticmethod
    def s_exp_decay(self, x_array, amp, tau):
        decay = amp * np.exp((-x_array)/tau)
        return decay

    @staticmethod
    def db_exp_decay(self, x_array, a_fast, tau1, a_slow, tau2):
        y = ((a_fast * np.exp((-x_array)/tau1))
            + (a_slow * np.exp((-x_array)/tau2)))
        return y

    @staticmethod
    def t_exp_decay(self, x_array, a_1, tau1, a_2, tau2, a_3, tau3):
        y = ((a_1 * np.exp((-x_array)/tau1))
            + (a_2 * np.exp((-x_array)/tau2))
            + (a_3 * np.exp((-x_array)/tau3)))
        return y


    @staticmethod
    def curve_fit_decay():
        pass

if __name__ == "__main__":
    curve_fit_decay_func()