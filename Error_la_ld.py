import numpy as np
from numpy import unravel_index
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
import math
import csv
 

def num_to_bin(n, l):   # l-bit 数字转换为二进制数组
    L = []
    bin_num = bin(n)[2:].zfill(l)
    for i in range(len(bin_num)):
        L.append(int(bin_num[i]))
    # print(n, L)
    return L


def bin_to_decimal(b):  # 二进制数组转换为小数
    d = 0
    for i, x in enumerate(b):
        # print("i, x: ", i, x)
        d += 2**(-i-1) * x
    return d


def gen_A(la):  # 生成 la-bit 的全部可能斜率，取值[0,1)，从小到大
    A = []
    for i in range(2**la):
        A.append(bin_to_decimal(num_to_bin(i, la)))
    return A

# la = 10
# print(gen_A(la))

def gen_D(ld):  # 生成 ld-bit 的全部可能斜率，取值[-1,0)，从小到大
    D = []
    for i in range(2**ld):
        D.append(-bin_to_decimal(num_to_bin(i, ld)))
    D.reverse()
    return D



######################## 函数 ########################

# GeLU, g(x)
def gx(x):
    return x/2 * erf(x/math.sqrt(2))
def gx_derivative(x):   # 计算拟合 g(x) 的一次函数的斜率
    return math.sqrt(2)*x*math.exp(-x**2/2)/(2*math.sqrt(math.pi)) + erf(math.sqrt(2)*x/2)/2
def gx_intercept(x):    # 计算截距
    return gx(x) - gx_derivative(x) * x

def error_a_d(C, a, d, start, end): # 计算误差，这里用均方误差
    x = np.linspace(start, end, 100)
    y_linear = a*x + d
    y_curve = gx(x)
    return np.mean((y_curve - y_linear) ** 2)

def error_a_d_aveULP(C, a, d, start, end): # 计算误差，这里用aveULP
    x = np.linspace(start, end, 100)
    y_linear = a*x + d
    y_curve = gx(x)
    return np.mean(np.abs(y_curve - y_linear))

def error_a_d_maxULP(C, a, d, start, end): # 计算误差，这里用maxULP
    x = np.linspace(start, end, 100)
    y_linear = a*x + d
    y_curve = gx(x)
    return np.max(np.abs(y_curve - y_linear))



def Error_slice(C, la, ld, start, end):

    A = gen_A(la)
    D = gen_D(ld)

    derivative_C_start = gx_derivative(start)
    derivative_C_end = gx_derivative(end)

    # print("derivative_C_start: ", derivative_C_start)
    # print("derivative_C_end: ", derivative_C_end)

    intercept_C_start = gx_intercept(start)
    intercept_C_end = gx_intercept(end)

    # print("intercept_C_start: ", intercept_C_start)
    # print("intercept_C_end: ", intercept_C_end)

    A_try, D_try = [], []
    for i in range(len(A)):
        if derivative_C_start < A[i] and A[i] < derivative_C_end:
            A_try.append(A[i])

    for i in range(len(D)):
        if intercept_C_start< D[i] and D[i] < intercept_C_end:
            D_try.append(D[i])

    if A_try == []:
        A_try.append(min(A, key=lambda x: abs(x - (derivative_C_start + derivative_C_end)/2)))
    if D_try == []:
        D_try.append(min(D, key=lambda x: abs(x - (intercept_C_start + intercept_C_end)/2)))

    # print("A_try: ", A_try)
    # print("D_try: ", D_try) # 不用D_try
    D_try = D   # 不用D_try

    e = [ [0 for i in range(len(D_try))] for j in range(len(A_try)) ]
    e = np.array(e, dtype=float)
    for i in range(len(A_try)):
        for j in range(len(D_try)):
            e[i][j] = error_a_d_maxULP(C, A_try[i], D_try[j], start, end)

    e_min =e.min()
    Ia,Id = unravel_index(e.argmin(), e.shape)
    a, d = A_try[Ia], D_try[Id]

    # print("e:\n ", e)
    # print("e_min: ", e_min)
    # print("Ia, Id: ", Ia, Id)
    # print("a, d: ", a, d)

    def draw():
        x = np.linspace(start, end, 100)
        y_curve = gx(x)
        y_linear = a*x + d
        plt.plot(x, y_curve, color = 'red')
        plt.plot(x, y_linear, color = 'blue')
        plt.grid(True)
        plt.show()
    # draw()

    return e_min, a, d


# C = 0
# la = 5
# ld = 12
# start, end = 2.94, 2.97
# Error_slice(C, la, ld, start, end)



def Error_all(C, la, ld, Start, End, N,s):    # 分成N份
    E_min, A, D = [], [], []
    x_curve = np.linspace(Start, End, 1000)
    y_curve = gx(x_curve)
    plt.plot(x_curve, y_curve, linewidth =0.2, color = 'red')

    for i in range(N):
        start = (Start + i/N * (End - Start))
        end = (start + 1/N * (End - Start))
        e_min_i, ai, di = Error_slice(C, la, ld, start, end)

        print(start, end)
        print("ai: ", (int)(ai*(2**la)))
        print("di: ", (int)(di*(2**ld)))
        print("ai: ", (ai))
        print("di: ", (di))
        print("e_min_i: ", e_min_i)
        print()

        E_min.append(e_min_i)
        A.append((int)(ai*(2**(la))))
        D.append((int)(di*(2**(ld))) % 2**(ld + 1))

        # draw
        x = np.linspace(start, end, 100)
        y_linear = ai*x + di
        plt.plot(x, y_linear, linewidth =0.2, color = 'blue')

    print("average: ", sum(E_min)/N)

    formatted_output = ', '.join(f'{{{a},{d}}}' for a, d in zip(A, D))
    print(formatted_output)

    csv_filename = 'la_ld_s.csv'

# 写入 CSV 文件
    with open(csv_filename, mode='a', newline='') as file:
        # writer = csv.writer(file)

        file.write(f'la={la},ld={ld},s={s}\n')

        file.write(f'std::vector<std::vector<uint64_t>> data = {{{formatted_output}}};\n')


    plt.grid(True)
    plt.show()
    plt.savefig("la-ld.png", dpi=300)

    


C = 0
la = 6
ld = 10
Start, End = 0, 4
s = 7
N = pow(2, s)
csv_filename = 'la_ld_s.csv'
with open(csv_filename, mode='w', newline='') as file:
    pass
for la in range(5, 13):  # la 从 5 到 12
    for lb in range(6, 13):  # lb 从 6 到 12
        for s in range(6, 8):  # s 从 6 到 7
            print(f"Executing Error_all with la={la}, lb={lb}, s={s}")
            Error_all(C, la, lb, Start, End, N, s) 



