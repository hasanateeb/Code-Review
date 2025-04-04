# Good Code Example (store in good_code/script1.py)
def add_numbers(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b

if __name__ == "__main__":
    result = add_numbers(5, 3)
    print(f"Sum: {result}")