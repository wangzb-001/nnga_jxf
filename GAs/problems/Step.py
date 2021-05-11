from problems.Problem_Abs import *


class Step(Problem_Abs):
    def Func(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        Z = np.sum(np.abs(X + 0.5) ** 2, axis=0)
        return Z
