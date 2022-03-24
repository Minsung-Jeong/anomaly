class Fourcal:
    # __init__ 은 생성자
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def add(self):
        result = self.first + self.second
        return result
    def mul(self):
        return self.first * self.second
    def div(self):
        return self.first / self.second

class babyFourCal(Fourcal):
    pass


cal = babyFourCal(1,2)

cal.div()