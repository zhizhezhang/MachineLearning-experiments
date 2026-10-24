import matplotlib.pyplot as plt

days = ["Monday", "Tuesday", "Friday", "Sunday"]
values = [1.5, 0.6, 7.8, 6.0]
colors = ["red", "green", "blue", "red"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(days, values, color=colors)
ax.set(title="Bar", xlabel="Day", ylabel="Number of students")
ax.tick_params(axis="x", labelsize=8)

fig.tight_layout()
plt.show()