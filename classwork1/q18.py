import sympy as sp
x = sp.symbols("x", real=True)
f = x**3 + 3*x**2 + 4
first = sp.diff(f, x)
second = sp.diff(first, x)
print("一阶导数：", first)
print("二阶导数：", second)
for point in sp.solve(first, x):
    curvature = second.subs(x, point)
    kind = "极大值" if curvature < 0 else "极小值"
print(f"x={point}，{kind}={f.subs(x, point)}")
