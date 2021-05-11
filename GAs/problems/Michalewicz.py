from problems.Problem_Abs import *


class Michalewicz(Problem_Abs):
    m = 10

    def __init__(self, m, valuable_num, Bound, pop_size, iterator_num, golbal_fv, accuracy):
        Problem_Abs.__init__(self, valuable_num, Bound, pop_size, iterator_num, golbal_fv, accuracy)
        self.m = m

    def Func(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        X = utils.pow_exe(self.Bound, X)
        d_range = np.arange(1, X.shape[0] + 1).T[:, np.newaxis]
        s1 = np.sin(X)
        xi = X ** 2
        si = d_range * xi
        si = si / np.pi
        s2 = np.sin(si) ** (2 * self.m)
        d = s1 * s2
        Z = -np.sum(d, axis=0)
        return Z


if __name__ == '__main__':
    michalewicz = Michalewicz(10, 2, [0.00, np.pi], 30, 20000, -1.8013, 0.0001)
    X = np.array([2.20, 1.57])
    print(michalewicz.Func(X))
