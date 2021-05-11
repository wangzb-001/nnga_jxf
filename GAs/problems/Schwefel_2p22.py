from problems.Problem_Abs import *

class Schwefel_2p22(Problem_Abs):
    def Func(self,X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        x1 = np.sum(np.abs(X), axis=0)
        x2 = np.cumprod(np.abs(X), axis=0)
        x2 = x2[-1, :]
        Z = x1 + x2
        return Z
