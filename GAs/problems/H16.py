from problems.Problem_Abs import *


class H16(Problem_Abs):

    def Func(self, X):
        X = utils.pow_exe(self.Bound, X)
        f = np.abs(X * np.sin(X) + 0.1 * X)
        Z = np.sum(f, axis=0)
        return Z
