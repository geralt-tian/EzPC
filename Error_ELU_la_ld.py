import numpy as np
from numpy import unravel_index
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
import math
import multiprocessing as mp

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

# la = 5
# print(gen_A(la))

def gen_D(ld):  # 生成 ld-bit 的全部可能斜率，取值[-1,0)，从小到大
    D = []
    for i in range(2**ld):
        D.append(-bin_to_decimal(num_to_bin(i, ld)))
    D.reverse()
    return D



######################## 函数 ########################

# GeLU, g(x)
# def gx(x):
#     # return x/2 * erf(x/1.141)
#     return 1/10 * x **2

# def gx_derivative(x):   # 计算拟合 g(x) 的一次函数的斜率
#     return math.sqrt(2)*x*math.exp(-x**2/2)/(2*math.sqrt(math.pi)) + erf(math.sqrt(2)*x/2)/2
# def gx_intercept(x):    # 计算截距
#     return gx(x) - gx_derivative(x) * x



# ELU
def gx(x):
    
    return (math.e**x - 1)

def gx_derivative(x):   # 计算拟合 g(x) 的一次函数的斜率
    return math.e**x

def gx_intercept(x):    # 计算截距
    return gx(x) - gx_derivative(x) * x






def error_a_d(C, a, d, start, end): # 计算误差，这里用均方误差
    x = np.linspace(start, end, 100)
    y_linear = a*x + d
    y_curve = gx(x)
    return np.max((y_curve - y_linear) ** 2)




def Error_slice(C, la, ld, start, end):

    A = gen_A(la)
    D = gen_D(ld)

    print("A: ", A)
    print("D: ", D)

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
    A_try = A
    D_try = D   # 不用D_try

    e = [ [0 for i in range(len(D_try))] for j in range(len(A_try)) ]
    e = np.array(e, dtype=float)
    for i in range(len(A_try)):
        for j in range(len(D_try)):
            e[i][j] = error_a_d(C, A_try[i], D_try[j], start, end)

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



# def Error_all(C, la, ld, Start, End, N,s):    # 分成N份
#     E_min, A, D = [], [], []
#     for i in range(N):
#         start = (Start + i/N * (End - Start))
#         end = (start + 1/N * (End - Start))
#         e_min_i, ai, di = Error_slice(C, la, ld, start, end)

#         print(start, end)
#         print("ai: ", ai)
#         print("di: ", di)
#         print("e_min_i: ", e_min_i)
#         print()

#         E_min.append(e_min_i)
#         A.append((int)(ai*(2**(la))))
#         D.append((int)(di*(2**(ld))) % 2**(ld + 1))

#         # draw
#         x = np.linspace(start, end, 100)
#         y_curve = gx(x)
#         y_linear = ai*x + di
#         plt.plot(x, y_linear, color = 'blue')


#     print("average: ", sum(E_min)/N)
#     formatted_output = ', '.join(f'{{{a},{d}}}' for a, d in zip(A, D))
#     print(formatted_output)
#     # print(set(A), len(set(A)))

#     csv_filename = 'elu_la_ld_s6.csv'

# # 写入 CSV 文件
#     with open(csv_filename, mode='a', newline='') as file:
#         # writer = csv.writer(file)

#         file.write(f'la={la+1},ld={ld+1},s={s}\n')

#         file.write(f'std::vector<std::vector<uint64_t>> data = {{{formatted_output}}};\n')

#     # x_curve = np.linspace(Start, End, 1000)
#     # y_curve = gx(x_curve)
#     # plt.plot(x_curve, y_curve, color = 'red')

#     # plt.grid(True)
#     # plt.show()

    


# # C = 0
# # la = 5
# # ld = 5
# # Start, End = -8, 0
# # N = 64
# # Error_all(C, la, ld, Start, End, N)


# C = 0
# la = 6
# ld = 10
# Start, End = -4, 0
# s = 6
# N = pow(2, s)
# csv_filename = 'la_ld_s.csv'
# with open(csv_filename, mode='w', newline='') as file:
#     pass
# for la in range(1, 13):  # la 从 5 到 12
#     for lb in range(1, 13):  # lb 从 6 到 12
#         for s in range(6, 7):  # s 从 6 到 7
#             print(f"Executing Error_all with la={la}, lb={lb}, s={s}")
#             Error_all(C, la, lb, Start, End, N, s) 


def Error_all_parallel(args):
    """将 Error_all 函数包装成支持多进程的函数"""
    C, la, ld, Start, End, N, s, csv_filename = args

    E_min, A, D = [], [], []
    for i in range(N):
        start = (Start + i / N * (End - Start))
        end = (start + 1 / N * (End - Start))
        e_min_i, ai, di = Error_slice(C, la, ld, start, end)

        print(start, end)
        print("ai: ", ai)
        print("di: ", di)
        print("e_min_i: ", e_min_i)
        print()

        E_min.append(e_min_i)
        A.append((int)(ai * (2 ** (la))))
        D.append((int)(di * (2 ** (ld))) % 2 ** (ld + 1))

    print("average: ", sum(E_min) / N)
    formatted_output = ', '.join(f'{{{a},{d}}}' for a, d in zip(A, D))
    print(formatted_output)

    # 将结果写入 CSV 文件
    with open(csv_filename, mode='a', newline='') as file:
        file.write(f'la={la + 1},ld={ld + 1},s={s}\n')
        file.write(f'std::vector<std::vector<uint64_t>> data = {{{formatted_output}}};\n')

    return f"Completed la={la}, ld={ld}, s={s}"

def main():
    C = 0
    Start, End = -8, 0
    s = 7
    N = pow(2, s)
    csv_filename = 'elu_la10_ld10_s6.csv'

    # 清空输出文件
    with open(csv_filename, mode='w', newline='') as file:
        pass

    # 构造参数列表
    tasks = []
    for la in range(1, 13):  # la 从 1 到 12
        for lb in range(1, 13):  # lb 从 1 到 12
            for s in range(7, 8):  # s 从 6 到 7
                print(f"Preparing task for la={la}, lb={lb}, s={s}")
                tasks.append((C, la, lb, Start, End, N, s, csv_filename))

    # 使用多进程运行
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(Error_all_parallel, tasks)

    # 输出所有完成的任务
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
