import matplotlib.pyplot as plt
labels = ["First", "Second", "Three"]
fig, ax = plt.subplots(figsize=(6, 4))
wedges, _, _ = ax.pie(
    [60, 30, 10], labels=labels, autopct="%1.1f%%",
    colors=["red", "yellowgreen", "lightskyblue"],
    explode=[0.05, 0, 0], startangle=90)
ax.legend(wedges, labels, loc="upper right",
          bbox_to_anchor=(1.22, 1.05))
ax.axis("equal")
fig.tight_layout()
plt.show()
