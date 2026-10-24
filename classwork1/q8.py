class Adder:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        return self.a + self.b

calculator = Adder(12, 8)
print("两数之和：", calculator.add())
