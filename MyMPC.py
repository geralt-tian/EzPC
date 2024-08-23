

import numpy as np
import math
import random
import matplotlib.pyplot as plt


def share(x, N):
    x0 = random.randint(0,N)
    x1 = (x - x0) % N
    return x0, x1


def unsign2sign(x, N):
    if x >= N/2:
        return x - N
    return x

def encode_real2ring(x, N, precision=12):
    return x * (2**precision) % N


def get_plaintext(x0, x1, N, precision=12):
    x = (x0 + x1) % N
    # print("000: ", x0, x1, x, N)
    res = x
    if x >= N/2:
        res = x - N
    return res / 2**precision


# TODO 处理符号
def Mul_uniform(x0, x1, y0, y1, l):
    x = (x0 + x1) % 2**l
    y = (y0 + y1) % 2**l
    return share(x * y, 2**l) 


# def Mul_non_uniform(x0, x1, y0, y1, lx, ly):
#     # print("x0, x1, lx: ", x0, x1, lx)
#     x = (x0 + x1) % 2**lx
#     y = (y0 + y1) % 2**ly
#     # print("xxx: ", x)
#     # print("yyy: ", y)
#     z = (x * y) % 2**(lx + ly)
#     # print("z: ", z)
#     msb = 0
#     if z >= 2**(lx + ly - 1):
#         msb = 1
#     sign_z = z - msb * 2**(lx + ly)
#     # print("msb: ", msb)
#     # print("z: ", z)
#     return share(sign_z, 2**(lx + ly))


def Mul_non_uniform(x0, x1, y0, y1, lx, ly, precision=12):

    x_plain = get_plaintext(x0, x1, 2**lx, precision)
    y_plain = get_plaintext(y0, y1, 2**ly, precision)
    z_plain = x_plain * y_plain

    print("x_plain: ", x_plain)
    print("y_plain: ", y_plain)
    print("z_plain: ", z_plain)

    return share(int(z_plain*2**precision), 2**(lx + ly))



def B2A(x0, x1, N):
    if (x0 + x1) == 0:
        return share(0, N)
    return share(1, N)

def wrap(x0, x1, N):
    wB = int(x0 + x1 >= N)
    return share(wB, 2)


def DReLU(x0, x1, N):
    x = (x0 + x1) % N
    if x < N/2:
        return share(1, 2)
    return share(0, 2)


def MUX(x0, x1, b, N):
    x = (x0 + x1) % N
    if b == 0:
        return share(0, N)
    return share(x, N)


def TrunReduce_logical(x0, x1, s, l, error=False):   
    """
    1. 逻辑右移
    2. 输出Z_{2**(l-s)}上的元素
    """
    u0, v0 = x0 >> s,  x0 % 2**s
    u1, v1 = x1 >> s,  x1 % 2**s

    w0, w1 = wrap(v0, v1, 2**s)
    w = (w0 + w1) % 2

    if error:
        t = (u0 + u1) % 2**(l-s)
    else:
        t = (u0 + u1 + w) % 2**(l-s)
    return share(t, 2**(l - s))


def LUT(T, i0, i1, Ni, N):
    i = (i0 + i1) % Ni
    return T[i]
    # return share(T[i], N)


def SExt_my(x0, x1, m, n):  # 可实现有符号的extension
    """
    1.可实现有符号的extension, 从Z_{2^m}扩张到Z_{2^n}
    2.要求|x| < 2^{m-2}
    """
    M = 2**m
    x = (x0 + x1) % M
    assert 0 <= x < M/4 or 3*M/4 <= x < M

    def area_div4(x0, x1, N):    # 判断点的位置
        a = int(x0 < N/4) * int(x1 < N/4)
        d = int(x0 > 3*N/4) * int(x1 > 3*N/4)
        w = 1 - a + d # w = 0,1,2, 表示(x0,x1)位于区域 A, B+C, D
        return w

    w = area_div4(x0, x1, 2**m)
    res = (x0 + x1 + 2**n - (2**m)*w) % 2**n 
    return share(res, 2**n)


# b0 + b1 = b = DReLU(x)    SirNN中有另一种等价实现
def DReLU2Wrap(x0, x1, b0, b1, N): 
    D0, D1 = DReLU(x0, x1, N)
    D = (D0 + D1) % 2

    m0 = int(x0 < N/2)
    m1 = int(x1 < N/2)

    I = m0 * m1
    # J = (m0 ^ 1) * (m1 ^ 1)
    J = m0 * m1 ^ m0 ^ m1 ^ 1

    # res = ((D ^ 1) * J) ^ (D * (I ^ 1))
    # res = (D * (I^J)) ^ D ^ J

    res = D * (m0 ^ m1) ^ J

    print("D, I, J: ",D, I, J)
    print("res: ", res)
    return res



def test_DReLU2Wrap(l=6):
    for x0 in range(2**l):
        for x1 in range(2**l):
            x = (x0 + x1) % (2**l)
            b0, b1 = wrap(x0, x1, 2**l)
            b = (b0 + b1) % 2
            res = DReLU2Wrap(x0, x1, b0, b1, 2**l)

            print("x0, x1, x: ", x0, x1, x)
            print("b0, b1, b: ", b0, b1, b)
            print("b: ", b)
            print("res:", res)
            assert (b == res)
            print(b == res)
            print()

# test_DReLU2Wrap(l=6)



# 此除法要求 |x| < 2^{l-2}
def division_map(x0, x1, divisor, N, error=False, test=False):

    def div0(x0, d):
        return int(x0/d) 

    def div1(x1, d):
        return N - int((N - x1)/d)
        
    def div1_D(x1, d):
        return (2*N - int((2*N - x1)/d)) % N
 
    xx0 = (x0 - int(N/4) ) % N
    xx1 = (x1 ) % N
    xx = (xx0 + xx1) % N

    xx0_tail = (xx0 + int(N/4) ) % divisor
    xx1_tail_1 =  ((N - xx1) % divisor)
    xx1_tail_2 =  ((2*N - xx1) % divisor)

    xx0_tail = round(xx0_tail, 10) % divisor
    xx1_tail_1 = round(xx1_tail_1, 10) % divisor 
    xx1_tail_2 = round(xx1_tail_2, 10) % divisor 

    z1_1 = div1(xx1, divisor) 
    z1_2 = div1_D(xx1, divisor) 

    K = int(N/2 <= xx0 and N/2 <= xx1)
           
    z1 = (z1_2 - z1_1) * K + z1_1
    xx1_tail = (xx1_tail_2 - xx1_tail_1) * K + xx1_tail_1


    if error:
        d = 0                         # 带1-bit误差
    else:
        d = int(xx0_tail < xx1_tail)    # 无误差
        
    z0 = (div0(xx0 + int(N/4), divisor)) % N - d 
    z = (z0 + z1) % N

    # d = 1
    # print("d: ", d)
    # z = int(x0/divisor) + int(x1/divisor) - 2*int(N/divisor)
    # z = z % N
    # return z, 0, z, xx0, xx1, xx


    # z0 = int(x0/divisor)
    # if x0 + x1 < N:
    #     z1 = int(x1/divisor) + d
    # else:
    #     z1 = N - int((N-x1)/divisor) 
    # z = (z0 + z1) % N
    # return z0, z1, z, xx0, xx1, xx


    # z0 = int(x0/divisor)
    # if x0 + x1 < N:
    #     z1 = (N - int((N-x1)/divisor)  + d) % N
    # else:
    #     z1 = (2*N - int((2*N-x1)/divisor)) % N
    # z = (z0 + z1) % N
    # return z0, z1, z, xx0, xx1, xx


    if test:
        print("xx0_tail, xx1_tail: ", xx0_tail, xx1_tail)
        print("d: ", d)
        return z0, z1, z, xx0, xx1, xx
    else:
        return z0, z1, z



def division_msb(x0, x1, divisor, N):


    # ######## x, divisor 均为正数
    # w = int(x0 + x1 >= N)
    # x0_tail = (x0) % divisor
    # x1_tail = ((w*N - x1) % divisor)
    # d = int(x0_tail < x1_tail )
    # print("x0_tail, x1_tail, d: ", x0_tail, x1_tail, d)
    # # d = 0
    # z0 = int(x0/divisor) - d
    # if w == 0:
    #     z1 = int(x1/divisor)
    # else:
    #     z1 = N - int((N - x1) / divisor)
    
    # z = (z0 + z1) % N
    # return z0, z1, z


    ######## x 为负数
    w = int(x0 + x1 >= N)
    x0_tail = (x0) % divisor
    x1_tail = (((w + 1)*N - x1) % divisor)
    d = int(x0_tail < x1_tail )
    print("x0_tail, x1_tail, d: ", x0_tail, x1_tail, d)
    # d = 0
    z0 = int(x0/divisor) - d
    z1 = ((w+1)*N - int(((w + 1) * N - x1) / divisor))% N
    
    z = (z0 + z1) % N
    return z0, z1, z



def test_div_all_area(N=2**6, lx=3):
    def x_div_d(x, d):
        if x < N/2:
            return x/d
        else:
            return N - ((N - x)/d)
    divisor = 3
    for x0 in range(N):
        for x1 in range(N):
            x = (x0 + x1) % N
            if x < 2**lx or x > N - 2**lx:
                z0, z1, z = division_msb(x0, x1, divisor, N)
                print("x0, x1, x", x0, x1, x)
                print("z0, z1, z", z0, z1, z)

                div_plain = int(x_div_d(x, divisor))
                print("div_plain: ", div_plain)
                print("z: ", z)

                delta = z - div_plain
                print("delta: ", delta)
                print("delta % N: ", delta % N)
                print()

                if (delta == 0):
                    plt.scatter(x0, x1, s=10, color = 'blue')
                if (delta != 0):
                    if (delta % N == 1):
                        plt.scatter(x0, x1, s=10, color = 'red')
                    elif (delta % N == N - 1):
                        plt.scatter(x0, x1, s=10, color = 'orange')
                    else:
                        plt.scatter(x0, x1, s=5, color = 'yellow')
    
    plt.show()

# test_div_all_area(N=2**5, lx=3)




def interval(x0, x1, a, b, N, error):
    xa = (x0 - a) % N
    # 要求 |x - a| < 2^{l-2}
    # z0, z1, z, xx0, xx1, xx = division_map(xa, x1, b - a, N)
    z0, z1, z = division_map(xa, x1, b - a, N, error)
    if z == 0:
        return share(1, N)
    return share(0, N)



def Muti_interval(x0, x1, a, s, N, error):
    """
    初值为a, 公差为s的等差数列, 判断x落在第几个区间
    """
    xa = (x0 - a) % N
    # 要求 |x - a| < 2^{l-2}
    z0, z1, z, xx0, xx1, xx = division_map(xa, x1, s, N, error)
    return z









################# test #################


def test_TrunReduce(l=6, lx=3, s=2):
    for x0 in range(2**l):
        for x1 in range(2**l):
            x = (x0 + x1) % (2**l)
            t_plain = (x >> s) % 2**(l-s)
            t0, t1 = TrunReduce_logical(x0, x1, s, l)
            t_ss = (t0 + t1) % 2**(l-s)
            if t_plain != t_ss:
                print("x0, x1, x: ", x0, x1, x)
                print("t_plain: ", t_plain)
                print("t_ss: ", t_ss)
                print()

# test_TrunReduce(l=6, lx=3, s=2)



def test_div_all_area(N=2**6, lx=3):
    def x_div_d(x, d):
        if x < N/2:
            return x/d
        else:
            return N - ((N - x)/d)

    divisor = 3
    for x0 in range(N):
        for x1 in range(N):
            x = (x0 + x1) % N
            if x < 2**lx or x > N - 2**lx:
                z0, z1, z, xx0, xx1, xx = division_map(x0, x1, divisor, N, error=False, test=True)
                
                print("x0, x1, x", x0, x1, x)
                print("xx0, xx1, xx", xx0, xx1, xx)
                print("z0, z1, z", z0, z1, z)

                div_plain = int(x_div_d(x, divisor))
                print("div_plain: ", div_plain)
                print("z: ", z)

                delta = z - div_plain
                print("delta: ", delta)
                print("delta % N: ", delta % N)
                print()

                if (delta == 0):
                    plt.scatter(x0, x1, s=10, color = 'blue')
                if (delta != 0):
                    if (delta % N == 1):
                        plt.scatter(x0, x1, s=10, color = 'red')
                    elif (delta % N == N - 1):
                        plt.scatter(x0, x1, s=10, color = 'orange')
                    else:
                        plt.scatter(x0, x1, s=5, color = 'yellow')
    
    plt.show()

# 测试
# test_div_all_area(N=2**5, lx=3)



def test_Muti_interval_all(N=2**6, lx=3):
    a, s = 1, 2
    for x0 in range(N):
        for x1 in range(N):
            x = (x0 + x1) % N
            if x < 2**lx or x > N - 2**lx:
                error = True
                z = Muti_interval(x0, x1, a, s, N, error)
                print("x0, x1, x: ", x0, x1, x)
                print("z: ", z)

                if x < N/2:
                    z_plain = int((x - a) / s)
                else:
                    z_plain = int((x - a) / s)
                print("z_plain: ", z_plain)
                print()


# test_Muti_interval_all(N=2**6, lx=3)




