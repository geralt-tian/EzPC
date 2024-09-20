
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.special import erf, erfinv

import MyMPC


precision = 12


def gx(x):
    return x/2 * erf(x/math.sqrt(2))

def erf_derivative(x):  # erf的导数
    return 2*np.exp(-x**2) / np.sqrt(np.pi)
 
def gx_derivative(x):   # 计算拟合 g(x) 的一次函数的斜率
    return erf(x/math.sqrt(2)) + x * (1/math.sqrt(2)) *erf_derivative(x/math.sqrt(2))

def gx_intercept(x):    # 计算截距
    return gx(x) - gx_derivative(x) * x


# tanh_part = std::tanh(0.7978845608 * a + 0.7978845608 * 0.044715 * std::pow(a, 3));
#     return 0.5 * a * (1 + tanh_part);

def draw_test():
    def draw_gx():
        # x = np.linspace(-5, 0, 1000)
        x = np.linspace(-5, 5, 1000)
        y = x/2 +  x/2 * erf(x/1.414)
        print(1.199951171875/2 +  1.199951171875/2 * erf(1.199951171875/1.414))
        tanh_part = np.tanh(0.7978845608 * x + 0.7978845608 * 0.044715 * x**3)

        yy =  0.5 * x * (1 + tanh_part)

        # y = x/2 * erf(x/1.141)
        # y =  erf(x/1.141)

        # y = (math.e**(2*x) - math.e**(-2*x) - 4*x) / (math.e**x + math.e**(-x))**2
        
        # # sigmoid
        # y = math.e**(-x) / (1 + math.e**(-x))**2
        # y = 0.5*(1-math.e**(-x))/(1+math.e**(-x)) - x*(math.e**(-x)/(1+math.e**(-x))**2)
        
        # e^x
        # y = 0.3585 * (x + 1.353)**2 + 0.344
        # plt.plot(x, y)
        # y = math.e**x

        # 绘制 gx 图像
        plt.plot(x, y,  color='red')
        plt.plot(x, yy, color='blue')
        plt.grid(True)

        # 保存图片为高分辨率 (300 dpi)
        plt.savefig("output_image.png", dpi=300)
        plt.show()

    def draw_a_d():
        # 确保所有代码块正确缩进
        x_values = []
        slope_values = []
        intercept_values = []
        k_d_pairs = []  # 用于存储{k, d}对
        # 计算斜率和截距
        for xx in range(0,128):
            x = xx / 32
            k = gx_derivative(x)  # 斜率
            d = gx_intercept(x)   # 截距

            k_d_pairs.append(f"{{{int((k)*(2**4))%(2**6)}, {int((d)*(2**4))%(2**6)}}}")
            # print("x: ", x)
            # print("k: ", k)
            # print("d: ", d)
            
            x_values.append(x)
            slope_values.append(int((k)*(2**4))%(2**6))
            intercept_values.append(int((d)*(2**4))%(2**6))
        print("{" + ", ".join(k_d_pairs) + "}")
        # 绘制散点图：红色表示斜率，蓝色表示截距
        plt.scatter(x_values, slope_values, s=1, color='red', label='Slope')
        plt.scatter(x_values, intercept_values, s=1, color='blue', label='Intercept')
        
        # 添加标题和标签
        plt.title('gx(x)')
        plt.xlabel('x')
        plt.ylabel('value')

        # 添加图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # # 保存图片为高分辨率 (300 dpi)
        # plt.savefig("output_image.png", dpi=300)

        # # 显示图像
        # plt.show()

    # 调用绘制函数
    draw_gx()
    # draw_a_d()
    
draw_test()





# 估值 g(x) = x * erf(x/sqrt(2))
def GELU_my(x):
    
    a, d = gx_derivative(abs(x)), gx_intercept(abs(x))  # LUT
    b = int(x < 0)  # l-bit 比较协议
    a = (-1)**b * a # la-bit MUX
    
    ax = a * x  # 秘密乘法
    z = ax + d + x/2

    print("x: ", x)
    print("a, d: ",a, d)
    if d >= 0 :
        print("d >= 0")
    print("z: ", z)

    I = int(-2.7 < x < 2.7) # 区间判断协议
    u = I * z # MUX
    v = (1-b) * (1-I) * x # MUX
    y = u + v
    return y



def test_GELU_my():
    for xx in range(-50, 50):    # x \in [-5, 5]
        x = xx/10

        y_real = x/2 * erf(x/1.141) + x/2
        y_my = GELU_my(x)
        print("x: ", x)
        print("y_real: ", y_real)
        print("y_my: ", y_my)
        print()

        plt.scatter(x, y_real, s=2, color = 'red')
        plt.scatter(x, y_my, s=1, color = 'blue')
    
    plt.show()
    plt.savefig("output_image1.png", dpi=300) 

draw_test()




