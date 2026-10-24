import numpy as np

total = 0
fact = 1
for i in range(1, 21):
    fact *= i
    total += fact
print("结果为：", total)