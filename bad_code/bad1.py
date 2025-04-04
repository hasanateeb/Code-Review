class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print("Woof!")

dog = Dog("Buddy")
dog.bark()
dog.meow()  # Error: This method does not exist
