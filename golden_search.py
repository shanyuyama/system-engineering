import numpy as np

def golden_section_search(f, direction=[4, -2], x0=[1, 2], tol=0.001):
    x0 = np.array(x0)
    direction = np.array(direction)
    a = x0
    b = x0 + 0.2*direction
    # print(f"a = {a}, b = {b}")
    x1 = b + 0.618*(a-b)
    x2 = a + 0.618*(b-a)
    f1 = f(x1)
    f2 = f(x2)
    while abs((f1-f2)/f1) > tol:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + 0.618*(b-a)
            f2 = f(x2)
            # print(f"a = {a}, b = {b}, x1 = {x1}, x2 = {x2}, f1 = {f1}, f2 = {f2}")
        elif f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + 0.618*(b-a)
            f1 = f(x1)
            # print(f"a = {a}, b = {b}, x1 = {x1}, x2 = {x2}, f1 = {f1}, f2 = {f2}")
    if f1 < f2:
        min_x = x1
    else:
        min_x = x2
    return min_x

# Example usage:
def example_function(x):
    return x[0]**4 - 2*x[0]**2*x[1] + x[1]**2 + x[0]**2 - 2*x[0] + 5


min_x = golden_section_search(example_function)
print(f"The minimum value of the function is at x = {min_x}")
print(example_function(min_x))
x = [1.4,1.8]
print(example_function(x))

