class User():
    def __init__(self, name, age):
        self.name = name
        self._age = age

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, n):
        self._age = n + 5


if __name__ == "__main__":
    user = User('aaa', 20)
    user.age = 5
    print(user.age)