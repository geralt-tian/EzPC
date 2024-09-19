import random
from functools import reduce

# 定义一个素数，作为有限域的大小（通常要比秘密大）
PRIME = 2**5 - 1

def poly_eval(coeffs, x):
    """计算多项式的值，f(x) = a_0 + a_1 * x + a_2 * x^2 + ..."""
    return sum([coeff * (x ** i) for i, coeff in enumerate(coeffs)]) % PRIME

def generate_shares(secret, num_shares, threshold):
    """
    生成份额
    secret: 秘密值
    num_shares: 生成的份额总数
    threshold: 恢复秘密所需的最少份额数
    """
    # 生成多项式的系数，f(0) = secret
    coeffs = [secret] + [random.randint(0, PRIME - 1) for _ in range(threshold - 1)]
    shares = [(x, poly_eval(coeffs, x)) for x in range(1, num_shares + 1)]
    return shares

def interpolate(shares, x=0):
    """
    拉格朗日插值法，重构秘密
    shares: 份额 [(x1, y1), (x2, y2), ...]
    x: 默认是0，用于重构秘密
    """
    def _lagrange_basis(i, x, x_s):
        xi, _ = x_s[i]
        terms = [(x - xj) * pow(xi - xj, -1, PRIME) % PRIME for j, (xj, _) in enumerate(x_s) if j != i]
        return reduce(lambda a, b: a * b % PRIME, terms, 1)

    secret = sum(y * _lagrange_basis(i, x, shares) % PRIME for i, (xi, y) in enumerate(shares)) % PRIME
    return secret

# 示例：生成份额并重构秘密
if __name__ == "__main__":
    secret = 12  # 你的秘密
    num_shares = 5      # 生成的份额数
    threshold = 2      # 恢复秘密所需的最少份额数

    # 生成份额
    shares = generate_shares(secret, num_shares, threshold)
    print("生成的份额:", shares)

    # 使用3个份额重构秘密
    selected_shares = shares[:threshold]  # 选取前3个份额
    recovered_secret = interpolate(selected_shares)
    print("重构的秘密:", recovered_secret)