class Person:
    """Represents a person with a name and age."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        """Greets with the person's name."""
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

person = Person("Alice", 30)
person.greet()
