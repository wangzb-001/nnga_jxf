from problems.Problem_Abs import *


class H9(Problem_Abs):
    def Func(self, X):
        X = utils.pow_exe(self.Bound, X)
        f = np.floor(X)
        Z = 6 * X.shape[0] + np.sum(f, axis=0)
        return Z
