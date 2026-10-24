import numpy as np
import matplotlib.pyplot as plt
class Circle:
    def __init__(self, radius):
        if radius <= 0:
            raise ValueError("半径必须为正数")
        self.radius = radius

    def draw(self):
        theta = np.linspace(0, 2 * np.pi, 400)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.radius * np.cos(theta),
                self.radius * np.sin(theta))
        ax.set(xlabel="x", ylabel="y", title=f"Circle r={self.radius:g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

Circle(float(input("输入圆的半径："))).draw()
