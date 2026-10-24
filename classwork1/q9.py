from pathlib import Path
s = input("输入小写字符串：")
result = s.upper()
path = Path("../uppercase.txt")
path.write_text(result, encoding="utf-8")
print("转换结果：", result)
print("文件内容：", path.read_text(encoding="utf-8"))
