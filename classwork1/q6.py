def swap(a, b):
    return b, a

x, y = 10, 20
print("交换前：", x, y)
x, y = swap(x, y)
print("交换后：", x, y)
