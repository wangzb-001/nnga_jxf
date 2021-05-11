from problems.Problem_Abs import *

class Sphere(Problem_Abs):
    def Func(self,X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        Z = np.sum(X ** 2, axis=0)
        return Z
