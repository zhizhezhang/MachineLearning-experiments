import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
A = np.array([[3, -2, 1], [2, -1, 1], [1, 1, 1]], float)
b = np.array([-1, -1, 2], float)
numeric = np.linalg.solve(A, b)
x, y, z = sp.symbols("x y z")
exact = sp.solve([3*x-2*y+z+1, 2*x-y+z+1, x+y+z-2],
                 (x, y, z))
symbolic = np.array([float(exact[v]) for v in (x, y, z)])
print("NumPy 数值解：", numeric)
print("SymPy 符号解：", exact)
print("最大残差：", np.max(np.abs(A @ numeric - b)))
assert np.allclose(numeric, symbolic)
positions = np.arange(3)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(positions - 0.18, numeric, 0.36, label="NumPy")
ax.bar(positions + 0.18, symbolic, 0.36, label="SymPy")
ax.set_xticks(positions, ["x", "y", "z"])
ax.set(ylabel="Solution", title="Comparison of two methods")
ax.legend()
fig.tight_layout()
plt.show()
