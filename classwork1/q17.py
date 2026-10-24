import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x = np.arange(10, dtype=float)
y = 2 * x + 5
noisy_y = y + rng.normal(0, 1, size=10)
i, j = rng.choice(10, size=2, replace=False)
k = (noisy_y[j] - noisy_y[i]) / (x[j] - x[i])
b = noisy_y[i] - k * x[i]
print(f"抽样点1：({x[i]:.0f}, {noisy_y[i]:.6f})")
print(f"抽样点2：({x[j]:.0f}, {noisy_y[j]:.6f})")
print(f"抽样直线：y = {k:.6f}x + {b:.6f}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, y, "o-", label="Original y=2x+5")
ax.scatter(x, noisy_y, marker="x", label="Noisy points")
ax.scatter(x[[i, j]], noisy_y[[i, j]], s=100,
           facecolors="none", edgecolors="red", label="Sampled points")
ax.plot(x, k*x+b, "--", label="Line through sample")
ax.set(xlabel="x", ylabel="y", title="Sampling noisy points")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
