from __future__ import annotations

import numpy as np
from scipy.interpolate import lagrange


class Parabola:
    def __init__(self, a, b, c):
        """f(x) = a * x^2 + b * x + c"""
        self.a = a
        self.b = b
        self.c = c

    def similarity(self, other: Parabola) -> float:
        return np.sqrt(np.square(self.a - other.a) + np.square(self.b - other.b) + np.square(self.c - other.c))

    def average(self, other: Parabola) -> Parabola:
        average_a = (self.a + other.a) / 2
        average_b = (self.b + other.b) / 2
        average_c = (self.c + other.c) / 2

        return Parabola(average_a, average_b, average_c)

    @classmethod
    def fit(cls, points: np.ndarray) -> Parabola:
        try:
            a, b, c = lagrange(points[0], points[1]).coeffs
        except ValueError:
            a, b, c = np.inf, np.inf, np.inf
        return Parabola(a, b, c)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.a * np.square(points) + self.b * points + self.c

def randomized_parabolic_hough_transform(binary_image: np.ndarray,
                                         iterations: int = 3,
                                         required_similarity: float = 0.5,
                                         required_hits: int = 3) -> [Parabola]:
    points_x, points_y = np.where(binary_image)
    points = np.stack([points_x, points_y], axis=0)
    num_points = len(points[0]) - 1
    accumulator = {}

    for _ in range(iterations):
        selected_indices = np.random.choice(np.arange(num_points), 3, replace=False)
        selected_points = points[:, selected_indices]

        new_candidate = Parabola.fit(selected_points)
        for known_candidate, value in accumulator.copy().items():
            if new_candidate.similarity(known_candidate) < required_similarity:
                del accumulator[known_candidate]
                average_candidate = new_candidate.average(known_candidate)
                accumulator[average_candidate] = value + 1
                break
        else:
            average_candidate = new_candidate
            accumulator[average_candidate] = 1

        if accumulator[average_candidate] > required_hits:
            print("HITS")
            return average_candidate

    print("not so many hits")
    print(accumulator)
    return max(accumulator, key=accumulator.get)
    # TODO: accumulate and return top K instead of top 1
