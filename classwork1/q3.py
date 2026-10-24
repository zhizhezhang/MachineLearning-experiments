import math
a, b, c = map(float, input("输入 a b c：").split())
if a == 0:
    if b != 0:
        print("一次方程的解：", -c / b)
    else:
        print("任意实数都是解" if c == 0 else "无解")
else:
    delta = b * b - 4 * a * c
    if delta < 0:
        print("无实数解")
    elif delta == 0:
        print("两个相等的实根：", -b / (2 * a))
    else:
        # 避免直接使用求根公式造成不必要的相减消去。
        q = -0.5 * (b + math.copysign(math.sqrt(delta), b))
        print("两个实根：", *sorted((q / a, c / q)))
