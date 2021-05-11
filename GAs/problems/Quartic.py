from problems.Problem_Abs import *


class Quartic(Problem_Abs):
    def Func(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        d_range = np.arange(1, X.shape[0] + 1).T[:, np.newaxis]
        Z = np.sum(d_range * (X ** 4), axis=0) + np.random.rand()
        return Z
