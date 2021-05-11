
from problems.Problem_Abs import *


class H14(Problem_Abs):
    def Func(self,X):
        X = utils.pow_exe(self.Bound, X)
        p = np.sum(X, axis=0)
        Z = np.exp(0.5 * p)
        Z = Z - 1
        Z = abs(Z)
        return Z
