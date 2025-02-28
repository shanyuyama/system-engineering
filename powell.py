import numpy as np

def powell_method(f, x0, epsilon=1e-6, max_iter=100):
    """
    鲍威尔法（Powell's method）实现无约束优化
    :param f: 目标函数
    :param x0: 初始点
    :param epsilon: 收敛精度
    :param max_iter: 最大迭代次数
    :return: 近似最优点和函数值
    """
    n = len(x0)
    directions = np.eye(n)  # 初始搜索方向（坐标轴方向）
    x = x0.copy()
    x_prev_cycle = x.copy()
    for _ in range(max_iter):
        fx_prev = f(x)
        max_delta = -np.inf
        max_idx = 0
        x_prev_cycle = x.copy()
        # 阶段1: 沿每个方向进行一维搜索
        for i in range(n):
            # 定义一维搜索函数（黄金分割法）
            def line_search(alpha):
                return f(x + alpha * directions[i])
            
            # 黄金分割法找最优步长
            a, b = bracket_minimum(line_search)
            alpha = golden_section(line_search, a, b, epsilon)
            print(f"第{i+1}维搜索: alpha = {alpha}")
            # 更新点并记录最大变化方向
            x_new = x + alpha * directions[i]
            delta = f(x) - f(x_new)
            if delta > max_delta:
                max_delta = delta
                max_idx = i
            x = x_new
        
        # 阶段2: 生成新方向并判断是否替换旧方向
        new_dir = x - x_prev_cycle
        if np.linalg.norm(new_dir) < epsilon:  # 新方向太小则终止
            break
        
        # 沿新方向进行一维搜索
        def line_search_new(alpha):
            return f(x + alpha * new_dir)
        
        a, b = bracket_minimum(line_search_new)
        alpha_new = golden_section(line_search_new, a, b, epsilon)
        x_new = x + alpha_new * new_dir
        
        # 判断是否替换旧方向
        if f(x_new) < f(x):
            x = x_new
            directions = np.delete(directions, max_idx, axis=0)  # 删除效果最差的方向
            directions = np.vstack([directions, new_dir / np.linalg.norm(new_dir)])  # 添加新方向
        
        if np.linalg.norm(x - x_prev_cycle) < epsilon:
            break
        x_prev_cycle = x.copy()
    
    return x, f(x)

# 辅助函数：黄金分割法实现一维搜索
def golden_section(f, a, b, tol=1e-6):
    ratio = (np.sqrt(5)-1)/2  # 0.618
    c = b - ratio*(b-a)
    d = a + ratio*(b-a)
    while abs(c-d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - ratio*(b-a)
        d = a + ratio*(b-a)
    return (a+b)/2

# 辅助函数：确定一维搜索区间
def bracket_minimum(f, x0=0, step=0.1, expand_factor=2.0, max_steps=100):
    a, fa = x0, f(x0)
    b, fb = a + step, f(a + step)
    if fb > fa:  # 反向搜索
        step = -step
        b, fb = a + step, f(a + step)
    for _ in range(max_steps):
        c = b + expand_factor*(b - a)
        fc = f(c)
        if fc > fb:
            return (a, c) if a < c else (c, a)
        a, fa = b, fb
        b, fb = c, fc
    return (a, c) if a < c else (c, a)

def objective(x):
    return 50 - 8*x[0] -2*x[1] + x[0]**2 + 2*x[1]**2 - x[0]*x[1]

# 初始点
x0 = np.array([1, 1])
result, f_val = powell_method(objective, x0)
print(f"最优解: x = {result}, f(x) = {f_val}")