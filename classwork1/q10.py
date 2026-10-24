import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), "--", label="cos(x)")
ax.set(xlabel="x (rad)", ylabel="y", title="Sine and cosine")
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
              ["0", "π/2", "π", "3π/2", "2π"])
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
