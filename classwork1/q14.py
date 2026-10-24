import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 2 * np.pi, 0.1)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(x, 2 * x + 5)
axes[0].set_title("Line")
axes[1].plot(x, np.sin(x), "--")
axes[1].set_title("Sin")
fig.tight_layout()
plt.show()
