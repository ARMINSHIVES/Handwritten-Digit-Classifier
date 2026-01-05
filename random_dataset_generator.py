"""
Random Dataset Generator

Generates random numbers for creating handwritten digit test datasets.
Users can write out the generated numbers by hand to create their own
test images for the digit classifier.
"""

import random


def generate_numbers(n, lb, ub):
    """Generate n random integers within a specified range.

    Args:
        n: Number of random integers to generate.
        lb: Lower bound (inclusive) of the random range.
        ub: Upper bound (inclusive) of the random range.

    Returns:
        List of n random integers between lb and ub.
    """
    numbers = []
    for i in range(n):
        numbers.append(random.randint(lb, ub))
    return numbers


if __name__ == "__main__":
    # Generate 10 random numbers between 0 and 99999
    # Users can write these numbers by hand to create test images
    n = 10
    numbers = generate_numbers(n, 0, 99999)
    for i in range(n):
        print(numbers[i])
