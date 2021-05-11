
from problems.Problem_Abs import *


class H6(Problem_Abs):
    def Func(self,X):
        X = utils.pow_exe(self.Bound, X)
        f = np.sin(X) + np.sin((2 * X / 3))
        Z = np.sum(f, axis=0)
        return Z
