s = input("输入一行字符：")
letters = spaces = digits = others = 0
for ch in s:
    if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
        letters += 1
    elif ch == ' ':
        spaces += 1
    elif '0' <= ch <= '9':
        digits += 1
    else:
        others += 1
print(f"英文字母：{letters}，空格：{spaces}")
print(f"数字：{digits}，其他字符：{others}")
