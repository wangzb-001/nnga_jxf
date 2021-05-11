from problems.Problem_Abs import *


class Schwefel(Problem_Abs):
    def Func(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        s1 = np.sin(np.sqrt(np.abs(X)))
        Z = np.sum(-X * s1, axis=0)
        Z = 418.9828 * self.valuable_num + Z
        return Z


if __name__ == '__main__':
    schwefel = Schwefel(3, [-500, 500], 30, 20000, 0, 0.001)
    X = np.array([1, 500, 1.])
    print(schwefel.Func(X))
