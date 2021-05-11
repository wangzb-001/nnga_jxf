from problems.Problem_Abs import *


class Sample(Problem_Abs):
    def Func(self, X):
        # if len(X.shape) == 1:
        #     X = X[:, np.newaxis]
        Y = X ** 2
        return Y.squeeze()
