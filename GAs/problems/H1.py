
from problems.Problem_Abs import *

class H1(Problem_Abs):
    def Func(self, X):
        # X[X == 0] = 0.00000001
        X = utils.pow_exe(self.Bound, X)
        f = np.sin(10 * X * np.pi) / (10 * X * np.pi)
        Z = np.sum(np.abs(f), axis=0)
        return Z
