
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt

import MyMPC


precision = 12


def unsign2sign(x, N):
    if x >= N/2:
        return x - N
    return x


# 区间 [0,8]
T_GELU_128_shift =[[-0.00038, -7e-05], [-0.000496, -6.3e-05], [-0.000645, -4.5e-05], [-0.000831, -1e-05], [-0.001065, 4.8e-05], [-0.001355, 0.000138], [-0.001712, 0.000271], [-0.002149, 0.000462], [-0.00268, 0.000726], [-0.003322, 0.001086], [-0.00409, 0.001565], [-0.005006, 0.002192], [-0.006088, 0.003002], [-0.007359, 0.004033], [-0.008842, 0.005328], [-0.010561, 0.006936], [-0.01254, 0.008911], [-0.014802, 0.01131], [-0.01737, 0.014196], [-0.020268, 0.017631], [-0.023513, 0.021683], [-0.027123, 0.026415], [-0.03111, 0.03189], [-0.035482, 0.038167], [-0.040238, 0.045294], [-0.045374, 0.053311], [-0.050874, 0.062239], [-0.056713, 0.072083], [-0.062855, 0.082822], [-0.069253, 0.094408], [-0.075845, 0.106759], [-0.082557, 0.119752], [-0.089298, 0.133224], [-0.095963, 0.14696], [-0.102431, 0.160693], [-0.108565, 0.174101], [-0.114213, 0.186799], [-0.119209, 0.198343], [-0.123374, 0.208226], [-0.126517, 0.215878], [-0.128437, 0.22067], [-0.128927, 0.22192], [-0.127778, 0.218897], [-0.124778, 0.210829], [-0.119721, 0.196918], [-0.112408, 0.176348], [-0.102654, 0.148304], [-0.090291, 0.111986], [-0.075172, 0.06663], [-0.057178, 0.011524], [-0.03622, -0.053968], [-0.012244, -0.130389], [0.014766, -0.218169], [0.044786, -0.317606], [0.077748, -0.428848], [0.113541, -0.551883], [0.152011, -0.686524], [0.192961, -0.832404], [0.236154, -0.988973], [0.281314, -1.155497], [0.328132, -1.331065], [0.376273, -1.514598], [0.425374, -1.704865], [0.47506, -1.9005], [0.52494, -2.100023], [0.574626, -2.30187], [0.623727, -2.504415], [0.671868, -2.706006], [0.718686, -2.904989], [0.763846, -3.099745], [0.807039, -3.288716], [0.847989, -3.470436], [0.886459, -3.643555], [0.922252, -3.806866], [0.955214, -3.959318], [0.985234, -4.100039], [1.012244, -4.228342], [1.03622, -4.34373], [1.057178, -4.445902], [1.075172, -4.534748], [1.090291, -4.610342], [1.102654, -4.67293], [1.112408, -4.722916], [1.119721, -4.760847], [1.124778, -4.787393], [1.127778, -4.803326], [1.128927, -4.809499], [1.128437, -4.806824], [1.126517, -4.796256], [1.123374, -4.778767], [1.119209, -4.75533], [1.114213, -4.726904], [1.108565, -4.694416], [1.102431, -4.658752], [1.095963, -4.620744], [1.089298, -4.58116], [1.082557, -4.540704], [1.075845, -4.500003], [1.069253, -4.459614], [1.062855, -4.420017], [1.056713, -4.381618], [1.050874, -4.34475], [1.045374, -4.309682], [1.040238, -4.276613], [1.035482, -4.245687], [1.03111, -4.216993], [1.027123, -4.190572], [1.023513, -4.166423], [1.020268, -4.144511], [1.01737, -4.124768], [1.014802, -4.107103], [1.01254, -4.091406], [1.010561, -4.077554], [1.008842, -4.065412], [1.007359, -4.054842], [1.006088, -4.045702], [1.005006, -4.037853], [1.00409, -4.031158], [1.003322, -4.025487], [1.00268, -4.020716], [1.002149, -4.01673], [1.001712, -4.013423], [1.001355, -4.010698], [1.001065, -4.00847], [1.000831, -4.006661], [1.000645, -4.005202], [1.000496, -4.004034], [1.00038, -4.003107]]

for i in range(len(T_GELU_128_shift)):
    for j in range(len(T_GELU_128_shift[0])):
        T_GELU_128_shift[i][j] = int(T_GELU_128_shift[i][j] * 2**precision)



def GELU_plain_shift(x):
    x = x - 4
    return 0.5 * x * (1 + math.tanh(0.7978845608 * x + 0.7978845608*0.044715*(x**3)))


def draw_GELU_plain_shift():
    for x in range(-600, 600):
        x = x / 100
        y = GELU_plain_shift(x)
    
        plt.scatter(x, y, s=1, color = 'blue')
    
    plt.show()

# draw_GELU_plain_shift()



def GELU_cipherGPT(x0, x1, l):
    
    alpha = 4
    alpha_p = alpha * 2**precision
    beta = 2 * alpha_p
    h = math.ceil(math.log(beta, 2))
    s = 7

    print("beta: ", beta)
    print("h: ", h)

    T = T_GELU_128_shift

    # x0 = (x0 + alpha_p) % 2**l

    x0_p = x0 % 2**h
    x1_p = x1 % 2**h
    x_p = (x0_p + x1_p) % 2**h

    print("x0_p, x1_p, x_p: ", x0_p, x1_p, x_p)

    error=False
    i0, i1 = MyMPC.TrunReduce_logical(x0_p, x1_p, h - s, h, error)
    i = (i0 + i1) % 2**s
    print("i: ", i)

    a, d = MyMPC.LUT(T, i0, i1, 2**s, N=0)
    print("a, d", a, d)

    la = l-h

    # 乘法，但cipherGPT中乘法有问题，x在2**h上，有些地方会被认成负值
    z_plain = a * (x_p / 2**precision) + d
    print("z_plain: ", z_plain)

    z0, z1 = MyMPC.share(z_plain, 2**l)

    # a0, a1 = MyMPC.share(a, 2**la)
    # print("a0, a1: ", a0, a1)
    # ax0, ax1 = MyMPC.Mul_non_uniform(a0, a1, x0_p, x1_p, la, h)
    # # 截断
    # # ax0, ax1 = int(ax0 / 2**precision), ax1 / 2**precision
    # ax = (ax0 + ax1) % 2**l
    # print("ax: ", (ax0 + ax1) % 2**l)
    
    # ax_plain = MyMPC.get_plaintext(ax0, ax1, N=2**l, precision=precision)
    # print("ax_plain: ", ax_plain)

    # z = ax + d
    # z0, z1 = MyMPC.share(z, 2**l)
    # print("z0, z1, z: ", z0, z1, z)
    # z_plain = MyMPC.get_plaintext(z0, z1, 2**l, precision=precision)
    # print("z_plain: ", z_plain)

    b0, b1 = MyMPC.DReLU((x0 - beta) % 2**l, x1, 2**l)
    b_p0, b_p1 = MyMPC.DReLU(x0, x1, 2**l)
    b = (b0 + b1) % 2
    b_p = (b_p0 + b_p1) % 2
    print("b, b': ", b, b_p)

    u0, u1 = MyMPC.MUX(z0, z1, b^b_p, 2**l)
    v0, v1 = MyMPC.MUX(x0, (x1 - 4*(2**precision) % 2**l), b, 2**l)

    y0 = (u0 + v0) % 2**l
    y1 = (u1 + v1) % 2**l
    y = (y0 + y1) % 2**l
    y_plain = MyMPC.get_plaintext(y0, y1, 2**l, precision=precision)

    # print("u: ", u)
    # print("v: ", v)
    # print("y: ", y)
    print()

    return y, y_plain


# l = 32
# x = 2.85
# print("x: ", x)
# print("GELU_plain_shift: ", GELU_plain_shift(x))

# x = int(x * 2**precision)
# x0, x1 = MyMPC.share(x, 2**l)
# print("x: ", x)

# print(GELU_cipherGPT(x0, x1, l))


def test_GELU_cipherGPT(l):
    for xx in range(0, 1000):    # x \in [0, 10]
        xx = xx/100
        x = int(xx * 2**precision)

        x0, x1 = MyMPC.share(x, 2**l)
        y_ring, y_real = GELU_cipherGPT(x0, x1, l)
        y_plain = GELU_plain_shift(xx)

        print("xx: ", xx)
        print("y_plain: ", y_plain)
        print("y_real: ", y_real)

        plt.scatter(xx, GELU_plain_shift(xx), s=1, color = 'red')
        plt.scatter(xx, y_real, s=1, color = 'blue')
    
    plt.show()


test_GELU_cipherGPT(l=32)



def GELU_our(x0, x1, l):
    
    alpha = 4
    alpha_p = alpha * 2**precision
    beta = 2 * alpha_p
    h = math.ceil(math.log(beta, 2))
    s = 7
    print("beta: ", beta)
    print("h: ", h)

    T = T_GELU_128_shift

    x0_p = x0 % 2**h
    x1_p = x1 % 2**h
    x_p = (x0_p + x1_p) % 2**h
    print("x0_p, x1_p, x_p: ", x0_p, x1_p, x_p)

    # error=True
    error=False
    i = MyMPC.TrunReduce_logical(x0_p, x1_p, h - s, h, error)
    print("i: ", i)

    a, d = MyMPC.LUT(T, i)
    la, ld = h, precision + 5 + 2
    a0, a1 = MyMPC.share(a, 2**la)
    d0, d1 = MyMPC.share(d, 2**ld)

    print("a, d", a, d)

    # # 先平凡乘再扩张的话，乘法可能会overflow
    # ax = MyMPC.Mul_non_uniform(a, x_p, h)
    # print("ax: ", ax)
    # ax = int(ax / 2**precision) # 截断
    # print("ax: ", ax)
    # ax0, ax1 = MyMPC.share(ax, 2**la)
    # ax = MyMPC.SExt_my(ax0, ax1, la, l)
    # print("ax: ", ax)


    ax = MyMPC.Mul_non_uniform(a, x_p, la, h)
    print("ax: ", ax)
    ax = int(ax / 2**precision) # 截断
    print("ax: ", ax)
    
    d = MyMPC.SExt_my(d0, d1, ld, l)

    ax0, ax1 = MyMPC.share(ax, 2**l)
    d0, d1 = MyMPC.share(d, 2**l)

    z0 = (ax0 + d0) % 2**l
    z1 = (ax1 + d1) % 2**l
    z = (z0 + z1) % 2**l

    print("z0, z1: ", z0, z1)
    print("z: ", z)

    # error = True
    error = False
    I = MyMPC.interval(x0, x1, 0, beta, 2**l, error)
    J = (MyMPC.DReLU((x0 - beta) % 2**l, x1, 2**l)) 
    print("I, J: ", I, J)

    u = MyMPC.MUX(z0, z1, I, 2**l)
    v = MyMPC.MUX(x0, (x1 - 4*(2**precision) % 2**l), J, 2**l)

    y = u + v
    y = unsign2sign(y, 2**l)

    print("u: ", u)
    print("v: ", v)
    print("y: ", y)
    print()

    return y, y/(2**precision)


# l = 32
# x = 5.38
# print("x: ", x)
# print("GELU_plain_shift: ", GELU_plain_shift(x))

# x = int(x * 2**precision)
# x0, x1 = MyMPC.share(x, 2**l)
# print("x: ", x)

# print(GELU_our(x0, x1, l))



def test_GELU_our(l):
    for xx in range(0, 1000):    # x \in [0, 10]
        xx = xx/100
        x = int(xx * 2**precision)

        x0, x1 = MyMPC.share(x, 2**l)
        y_ring, y_real = GELU_our(x0, x1, l)
        y_plain = GELU_plain_shift(xx)

        print("xx: ", xx)
        print("y_plain: ", y_plain)
        print("y_real: ", y_real)

        plt.scatter(xx, GELU_plain_shift(xx), s=1, color = 'red')
        plt.scatter(xx, y_real, s=1, color = 'blue')
    
    plt.show()


# test_GELU_our(l=32)

