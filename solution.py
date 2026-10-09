""" 
Project 2: Edge Detection
-----------------------------------------------------------------------


Usage: 
    python solution.py <img1 path> <img2 path> <img size> <kernel size>

Example:
    python solution.py pic1grey300.jpg pic2grey300.jpg 300 3

"""

import cv2
import numpy as np
import math
import sys


FILTER = 'sharpening_filter3x3.txt'

def apply_filter(img, m, n, img_filter):
    # This function convolves images using the specified filter

    if not validate_img(img, n):
        raise Exception(f"Invalid img size, expected: ({n}, {n}) got: {img.shape}")
    if not validate_img(img_filter, m):
        raise Exception(f"Invalid filter size, expected: ({m}, {m}) got: {img_filter.shape}")
    
    result = np.zeros((n, n), dtype=np.uint8)

    for y in range(m//2, n - m//2):
        for x in range(m//2, n - m//2):
            p_val = 0

            if not check_conv_boundary(y, x, m, n):
                p_val = convolve(img, y, x, img_filter, m)

            result[y, x] = p_val

    return result
            

def convolve(img, y, x, f, m):
    # Applies kernel to calculate value

    y -= m // 2
    x -= m // 2

    result = 0 
    for i in range(m):
        for j in range(m):
            result += f[i][j] * img[y + i, x + j]

    return result

def normalize(img_filter):
    min = img_filter.min()
    max = img_filter.max()
    normalized = (img_filter - min) / (max - min)

    return normalized

def check_conv_boundary(y, x, m, n):
    return (
        y < m // 2 or
        y >= n - m // 2 or
        x < m // 2 or
        x >= n - m // 2
    )

def g(k, s = 1):
    # Generates gaussian kernel based on the params
    kernel = np.zeros((k, k))
    radius = k // 2

    for i in range(k):
        for j in range(k):
            y = i - radius
            x = j - radius

            kernel[i, j] = math.exp(-(y ** 2 + x ** 2)/ (2 * s ** 2))

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def validate_img(img, s):
    return (img.shape[0] == s and img.shape[1] == s)


def main():
    if not len(sys.argv) == 5:
        raise ValueError(f"Invalid args\nUsage: python solution.py <img1 path> <img2 path> <img size> <kernel size>")

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    n = int(sys.argv[3])
    m = int(sys.argv[4])

    ### TASK 1 ###
    f1 = g(m, 1)    # gaussian blur
    img1_blur = apply_filter(img1.copy(), m, n, f1)
    cv2.imwrite('img1_blur.png', img1_blur)

    f2 = np.loadtxt(FILTER, dtype=np.float32)    # simple 3x3 sharpening kernel
    f2 = normalize(f2)
    img1_sharpened = apply_filter(img1.copy(), m, n, f2)
    cv2.imwrite('img1_sharpened.png', img1_sharpened)


if __name__ == '__main__':
    main()