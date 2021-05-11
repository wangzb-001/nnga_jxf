from problems.Problem_Abs import *


class Ackley(Problem_Abs):
    def Func(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m = X.shape[0]
        b = 0.2
        a = 20
        c = 2 * np.pi
        X = utils.pow_exe(self.Bound, X)
        x_sum = -b * np.sqrt(np.sum(X ** 2, axis=0) / m)
        cos_sum = np.exp(np.sum(np.cos(c * X), axis=0) / m)
        Z = a + np.exp(1) - a * np.exp(x_sum) - cos_sum
        return Z
