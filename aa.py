import math

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


popt_precision = 6
scale_factor = 2**12  # 选择缩放因子
modulus = 2**22  # L12的环


'''
class
'''
class PieceConfig:
    def __init__(self, x_start=1, x_end=10, point_count=10000, piece_count=10):
        self.x_start = x_start
        self.x_end = x_end
        self.point_count = point_count
        self.piece_count = piece_count
        self.piece_interval = math.ceil(self.point_count / self.piece_count)

class CurveConfig:
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    @classmethod
    def init_default_curve(cls):
        CurveConfig.SIN = CurveConfig( lambda x: np.sin(x) )
        # CurveConfig.GELU = CurveConfig( lambda x: 0.5 * x * (1 + math.tanh(0.7978845608 * x + 0.7978845608*0.044715*(x**3))) )
        CurveConfig.GELU = CurveConfig( lambda x: 0.5 * (x-4) * (1 + math.tanh(0.7978845608 * (x-4) + 0.7978845608*0.044715*((x-4)**3))) )

        CurveConfig.RELU = CurveConfig( lambda x: np.where(x < 0, 0, x) )
        CurveConfig.EXP = CurveConfig( lambda x: np.e**(x))
        CurveConfig.SIGMOID = CurveConfig( lambda x: np.divide(1, 1 + np.e**(-x)) )
        CurveConfig.TANH = CurveConfig( lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) )
        CurveConfig.LEAKY_RELU = CurveConfig( lambda x: np.where(x <= 0, 1 * (np.exp(x) - 1), x) )
        CurveConfig.Inverse = CurveConfig( lambda x: (1/x) )

        CurveConfig.ONCE = CurveConfig( lambda x, a, b: a * x + b )
        CurveConfig.TWICE = CurveConfig( lambda x, a, b, c: a * x ** 2 + b * x + c )
        CurveConfig.FOURTH = CurveConfig( lambda x, a, b, c, d: a * x**4 + b * x**2 + c * x + d )


def float_to_ring(value, scale_factor, modulus):
    """
    将浮点数转换为整型，并映射到模 modulus 的整数环上
    """
    scaled_value = int(np.round(value * scale_factor))
    ring_value = scaled_value % modulus
    return ring_value
'''
function
'''
def piece_fit(target_curve_config, estimate_curve_config, piece_config:PieceConfig):
    popt_list = []
    ulp_list = []
    x_total = np.linspace(piece_config.x_start, piece_config.x_end, piece_config.point_count)

    Diff = np.array([])
    MSE = []

    for i in range(piece_config.piece_count):
        x_piece = x_total[i*piece_config.piece_interval : (i+1)*piece_config.piece_interval]
        y_piece_target = np.vectorize(target_curve_config.func)(x_piece)

        # 计算拟合曲线系数
        popt, pcov = curve_fit(estimate_curve_config.func, x_piece, y_piece_target)
        popt = np.vectorize(lambda popt_i: round(popt_i, popt_precision))(popt)

        # 将浮点数转换为整数环上的整数
        popt_ring = [float_to_ring(p, scale_factor, modulus) for p in popt]
        popt_list.append(popt_ring)

        print(f'\npiece: {i}, popt (float): {popt}')
        print(f'popt (ring): {popt_ring}')

        # 绘制原始数据和拟合曲线
        y_piece_estimate = np.vectorize(estimate_curve_config.func)(x_piece, *popt)

        plt.scatter(x_piece, y_piece_target, s=2)
        plt.plot(x_piece, y_piece_estimate, color='red')

        # 均方误差
        mse = mean_squared_error(y_piece_target, y_piece_estimate)
        MSE.append(mse)

    print()
    print("popt_precision: ", popt_precision)
    MSE = np.array(MSE)
    print("total MSE: ", np.mean(MSE))
    

    plt.show()
    plt.savefig("output_image.png", dpi=300) 
    return popt_list


def ULP(target, estimate):
    def count_decimal_places(n):
        split_n = str(n).split('.')
        return len(split_n[1]) if len(split_n) > 1 else 0

    decimal_places = np.vectorize(count_decimal_places)(estimate)

    delta = abs(target - estimate)
    # print("delta: ", delta)
    ulp = delta * (10 ** decimal_places)
    # print("ulp: ", ulp)
    avg_ulp = sum(ulp) / len(ulp)

    # print("delta: ", delta)
    # print("ULP: ", ulp)
    print("avg ULP: ", avg_ulp)

    return avg_ulp


'''
main
'''
if __name__ == '__main__':
    CurveConfig.init_default_curve()

    # piece_config = PieceConfig(x_start=0, x_end=8, point_count=1280, piece_count=128)
    piece_config = PieceConfig(x_start=0, x_end=8, point_count=1280, piece_count=256)
    

    # popt_list = piece_fit(CurveConfig.SIN, CurveConfig.ONCE, piece_config)
    # popt_list = piece_fit(CurveConfig.SIN, CurveConfig.TWICE, piece_config)
    # popt_list = piece_fit(CurveConfig.SIN, CurveConfig.FOURTH, piece_config)

    # print("piece_count: ", piece_count)
    popt_list = piece_fit(CurveConfig.GELU, CurveConfig.ONCE, piece_config)
    # popt_list = piece_fit(CurveConfig.GELU, CurveConfig.TWICE, piece_config)

    # popt_list = piece_fit(CurveConfig.EXP, CurveConfig.ONCE, piece_config)

    # popt_list = piece_fit(CurveConfig.Inverse, CurveConfig.ONCE, piece_config)
    # popt_list = piece_fit(CurveConfig.Inverse, CurveConfig.TWICE, piece_config)
# 

    # popt_list = piece_fit(CurveConfig.RELU, CurveConfig.TWICE, piece_config)

    # popt_list = piece_fit(CurveConfig.SIGMOID, CurveConfig.ONCE, piece_config)
    # popt_list = piece_fit(CurveConfig.SIGMOID, CurveConfig.TWICE, piece_config)
    # popt_list = piece_fit(CurveConfig.TANH, CurveConfig.TWICE, piece_config)
    # popt_list = piece_fit(CurveConfig.LEAKY_RELU, CurveConfig.TWICE, piece_config)

    # 输出表
    T = []
    for i in range(len(popt_list)):
        T.append(list(popt_list[i]))
    print(T)


    # print(*popt_list, sep='\n')
    # print(*popt_list)


    # y_test = np.array([1,2,3])
    # y_pred = np.array([2,2,3])

    # mse = mean_squared_error(y_test, y_pred)
    # print('mse:', mse)
