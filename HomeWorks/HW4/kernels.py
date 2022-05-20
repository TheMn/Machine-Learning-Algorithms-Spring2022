import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma):
    return np.exp(-(np.linalg.norm(x-y)**2) / (2 * (sigma ** 2)))