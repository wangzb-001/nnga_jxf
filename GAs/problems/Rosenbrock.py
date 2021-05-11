from problems.Problem_Abs import *

class Rosenbrock(Problem_Abs):
    def Func(self,X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        x1 = X[1:]
        x2 = X[:-1]
        Z = np.sum(100 * (x1 - x2 ** 2.0) ** 2 + (x2 - 1) ** 2.0, axis=0)
        return Z
