from problems.Problem_Abs import *


class Rastrigin(Problem_Abs):
    def Func(self, X):
        X = utils.pow_exe(self.Bound, X)
        Z = np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=0)
        Z = Z + 10 * self.valuable_num
        return Z


if __name__ == '__main__':
    rastrigin = Rastrigin(30, [-5.12, 5.12], 30, 30000, 0, 0.001)
    X = np.zeros(shape=30)
    print(rastrigin.Func(X))
