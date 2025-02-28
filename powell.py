import numpy as np

def powell_method(f, x0, epsilon=1e-6, max_iter=100):
    n = len(x0)
    directions = np.eye(n)  # 初始搜索方向（坐标轴方向）
    x = x0.copy()
    x_prev_cycle = x.copy()  # 初始化前一阶段初始点
    
    for _ in range(max_iter):
        fx_prev = f(x_prev_cycle)
        max_delta = -np.inf
        max_idx = 0
        
        # 阶段1: 沿每个方向进行一维搜索
        x_current = x_prev_cycle.copy()
        f_values = [fx_prev]  # 记录各点函数值
        
        for i in range(n):
            # 一维搜索函数
            def line_search(alpha):
                return f(x_current + alpha * directions[i])
            
            # 确定搜索区间并优化
            a, b = bracket_minimum(line_search)
            alpha = golden_section(line_search, a, b, epsilon)
            x_current += alpha * directions[i]
            f_current = f(x_current)
            f_values.append(f_current)
            
            # 记录最大delta及其方向索引
            delta = f_values[-2] - f_current
            if delta > max_delta:
                max_delta = delta
                max_idx = i
        
        # 阶段2: 判断是否生成新方向
        f1 = fx_prev                  # f(x^{k,0})
        f2 = f_values[-1]             # f(x^{k,n})
        f3 = f(2*x_current - x_prev_cycle)  # f(2x^{k,n} - x^{k,0})
        
        # 判断是否允许生成新方向（Powell条件）
        condition1 = f3 >= f1
        left = (f1 - 2*f2 + f3) * (f1 - f2 - max_delta)**2
        right = 0.5 * max_delta * (f1 - f3)**2
        condition2 = left >= right
        
        if condition1 or condition2:
            # 不生成新方向，保留原有方向
            x = x_current
            x_prev_cycle = x.copy()
            continue
        else:
            # 生成新方向并进行一维搜索
            new_dir = x_current - x_prev_cycle
            new_dir_norm = new_dir / np.linalg.norm(new_dir)
            
            def line_search_new(alpha):
                return f(x_current + alpha * new_dir)
            
            a, b = bracket_minimum(line_search_new)
            alpha_new = golden_section(line_search_new, a, b, epsilon)
            x_new = x_current + alpha_new * new_dir
            
            # 更新方向和点
            if f(x_new) < f(x_current):
                directions = np.delete(directions, max_idx, axis=0)
                directions = np.vstack([directions, new_dir_norm])
                x = x_new
                x_prev_cycle = x.copy()
            else:
                x = x_current
                x_prev_cycle = x.copy()
        
        # 收敛判断
        if np.linalg.norm(x - x_prev_cycle) < epsilon:
            break
    
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
x0 = np.array([1.0, 1.0])
result, f_val = powell_method(objective, x0)
print(f"最优解: x = {result}, f(x) = {f_val}")