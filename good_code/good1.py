import math

def calculate_area(radius):
    """Calculates the area of a circle."""
    if radius <= 0:
        raise ValueError("Radius must be positive")
    return math.pi * radius ** 2

try:
    area = calculate_area(5)
    print("Area:", area)
except ValueError as e:
    print(e)
