import numpy as np
import sympy as sp

# 定义变量和参数
x1, x2, x3 = sp.symbols('x1 x2 x3')
a = [sp.symbols(f'a{i}') for i in range(1,11)]  # a1-a10
b = [sp.symbols(f'b{i}') for i in range(1,11)]  # b1-b10

# 构造函数集
F = []
for i in range(10):
    f_i = x2 * a[i]**x1 + x3 - b[i]
    F.append(f_i)

# 计算雅可比矩阵
J = sp.Matrix([[sp.diff(f, var) for var in (x1, x2, x3)] for f in F])

# 显示矩阵结构
print("雅可比矩阵结构：")
sp.pprint(J)

params = [
    (4658, 0.0399),
    (5820, 0.0386),
    (6525, 0.0368),
    (7400, 0.0362),
    (9045, 0.0349),
    (10350, 0.0339),
    (11050, 0.0341),
    (11820, 0.0323),
    (12850, 0.0324),
    (13840, 0.0321)
]

# 构建代入字典（同时替换a和b的符号）
subs_dict = {}
for i in range(10):
    subs_dict[a[i]] = params[i][0]  # 代入a系数
    subs_dict[b[i]] = params[i][1]  # 代入b系数

# 数值计算雅可比矩阵
J_num = J.subs(subs_dict).evalf()

print("\n数值化后的雅可比矩阵示例：")
np.set_printoptions(precision=3)
print(J_num)

x_values = {x1: 0.5, x2: -0.1, x3: 0.001}
J_at_x = J_num.subs(x_values).evalf()

print("\n在x=[0.5, -0.1, 0.001]处的雅可比矩阵：")
print(np.array(J_at_x.tolist(), dtype=float))

x_values = {x1: 0.5, x2: -0.1, x3: 0.001}

# 将x的数值合并到替换字典中
full_subs = {**subs_dict, **x_values}

# 计算F(x)的数值
F_num = [f.subs(full_subs).evalf() for f in F]
print(type(F_num))
print("\nF(x) 的数值结果（保留6位小数）：")
for i, val in enumerate(F_num, 1):
    print(f"f{i}(x) = {float(val):.6f}")

# 将F_num转换为SymPy列向量矩阵
F_num_matrix = sp.Matrix(F_num)

# 使用SymPy单位矩阵替代numpy的eye
P = (J_at_x.T * J_at_x + sp.eye(3)).inv() * J_at_x.T * F_num_matrix

print("\nP(x) 的数值结果：")
sp.pprint(P.evalf(6))  # 保留6位有效数字