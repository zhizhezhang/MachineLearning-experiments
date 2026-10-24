import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
values = rng.random(50)
print("随机数数量：", len(values))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, 51), values, "o-", markersize=4)
ax.set(xlabel="Index", ylabel="Value", title="50 random values",
       ylim=(0, 1))
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
