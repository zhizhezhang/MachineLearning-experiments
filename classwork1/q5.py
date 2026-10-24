a = int(input("输入数字 a（0到9）："))
n = int(input("输入最大长度 n（正整数）："))
if not 0 <= a <= 9 or n < 1:
    raise ValueError("a 必须为0到9，n 必须为正整数")
term = total = 0
for _ in range(n):
    term = term * 10 + a
    total += term
print("和为：", total)
