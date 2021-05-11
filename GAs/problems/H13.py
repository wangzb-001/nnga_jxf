
from problems.Problem_Abs import *


class H13(Problem_Abs):
    def Func(self,X):
        X = utils.pow_exe(self.Bound, X)
        Z = np.max(np.abs(X),axis=0)
        return Z
