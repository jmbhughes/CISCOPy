import numpy as np

from ciscopy.hough import Parabola, randomized_parabolic_hough_transform


def test_hough_perfect_scenario():
    x = np.linspace(0, 99, 100)
    p = Parabola(0.3, -5, 50)
    y = p(x)

    image = np.zeros((100, 100))
    kept = np.where((0 <= y) * (y < 100))
    image[np.round(x[kept]).astype(int), np.round(y[kept]).astype(int)] = 1

    result = randomized_parabolic_hough_transform(image,
                                                  required_similarity=0.5,
                                                  iterations=1000,
                                                  required_hits=50)
    print(result.a, result.b, result.c)
    # TODO: make a real comparison

def test_hough_noisy_scenario():
    x = np.linspace(0, 99, 100)
    p = Parabola(0.3, -5, 50)
    y = p(x)

    image = np.zeros((100, 100))
    kept = np.where((0 <= y) * (y < 100))
    image[np.round(x[kept]).astype(int), np.round(y[kept]).astype(int)] = 1

    num_noise_pixels = int(np.sum(image) * 1.5)
    image[np.random.randint(0, 100, num_noise_pixels), np.random.randint(0, 100, num_noise_pixels)] = 1

    result = randomized_parabolic_hough_transform(image,
                                                  required_similarity=1.0,
                                                  iterations=1000,
                                                  required_hits=50)
    print(result.a, result.b, result.c)
    # TODO: make a real comparison
