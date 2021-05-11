from problems.Problem_Abs import *


class H5(Problem_Abs):
    def Func(self, X):
        X = utils.pow_exe(self.Bound, X)
        f = X * np.sin(10 * np.pi * X)
        Z = -np.sum(f, axis=0)
        return Z
