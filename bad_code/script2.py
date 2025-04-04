def factorial(n):
    """Function lacks type hints and base case validation."""
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial("5"))  # Incorrect input type (should be an int)
