import numpy as np
import scipy.special as sc_special


def size(self, X):
    s = 0
    for i in range(self.test_func.pop_size):
        p = self.pop[:, i]
        b = np.abs((X - p)) < self.test_func.accuracy
        bb = b.all()
        if bb:
            s += 1
    return s


def pow_exe(bound, X, p=1):
    return X
    # def sigmoid(X):
    #     f = 1 / (1 + np.exp(-X))
    #     return f
    #
    # def pow(X, p=2):
    #     return X ** p
    #
    # sx = pow(X, p)
    # f = bound[0] + (bound[1] - bound[0]) * sx
    # return f


# def test():
#     X =np.random.normal(0,1,10)
#     # X = np.zeros(10)
#     print(X)
#     Y = sigmoid_exe([-100, 100], X)
#     print(Y)

def func_levy(X_, ala=0.001):
    '''

    :param X:
    :param ala:
    :return:
    '''
    c = np.random.standard_cauchy(X_.shape[0])
    X = X_ + ala * c
    return X


def func_levy_v(x, ala=0.001):
    '''

    :param X:
    :param ala:
    :return:
    '''
    c = np.random.standard_cauchy(1)
    X = x + ala * c[0]
    return X


class Levy:
    pre_pos1 = None
    pre_pos2 = None

    @classmethod
    def levy_flight(cls, n, m, beta=1.5):
        """
        This function implements Levy's flight.
        ---------------------------------------------------
        Input parameters:
            n: Number of steps,（种群大小）
            m: Number of dimensions,问题维度
            beta: Power law index (note: 1 < beta < 2)
        Output:
            'n' levy steps in 'm' dimension
        """
        sigma_u = (sc_special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                sc_special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, (n, m))
        v = np.random.normal(0, sigma_v, (n, m))

        steps = u / ((np.abs(v)) ** (1 / beta))
        # if cls.pre_pos1 is not None and cls.pre_pos1.shape == steps.shape:
        #     steps = steps + cls.pre_pos1
        #     cls.pre_pos1 = steps
        # if cls.pre_pos2 is not None and cls.pre_pos2.shape == steps.shape:
        #     steps = steps + cls.pre_pos2
        #     cls.pre_pos2 = steps
        return steps


if __name__ == '__main__':
    x = 0
    for i in range(100):
        print(x)
        x = func_levy_v(x)
