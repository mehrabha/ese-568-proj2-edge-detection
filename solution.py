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

    for y in range(n):
        for x in range(n):
            if not check_conv_boundary(y, x, m, n):
                p_val = convolve(img, y, x, img_filter, m)
                result[y, x] = int(round(np.clip(p_val, 0, 255)))

    return result


def apply_separable_filter(img, m, n, img_filter):
    # This function convolves images using the separable filter for faster runtime

    if not validate_img(img, n):
        raise Exception(f"Invalid img size, expected: ({n}, {n}) got: {img.shape}")
    if m != len(img_filter):
        raise Exception(f"Invalid filter size, expected: {m} got: {len(img_filter)}")
    
    result1 = img.copy()

    # convolve the rows
    for y in range(n):
        for x in range(n):
            if not check_conv_boundary(y, x, m, n):
                p_val = convolve(img, y, x, img_filter, m, separable=True, axis=1)
                result1[y, x] = p_val

    result2 = result1.copy()
    # convolve the cols
    for y in range(n):
        for x in range(n):
            if not check_conv_boundary(y, x, m, n):
                p_val = convolve(result1, y, x, img_filter, m, separable=True, axis=0)
                result2[y, x] = int(round(np.clip(p_val, 0, 255)))

    return result2


def convolve(img, y, x, f, m, separable = False, axis = 0):
    # Applies kernel to calculate value
    result = 0 

    if not separable:
        y -= m // 2
        x -= m // 2
        
        for i in range(m):
            for j in range(m):
                result += f[i][j] * img[y + i, x + j]
    else:
        if axis == 0:
            y -= m // 2

            for i in range(m):
                result += f[i] * img[y + i, x]
        elif axis == 1:
            x -= m // 2

            for i in range(m):
                result += f[i] * img[y, x + i]

    return result

def calculate_edges(img, size, thresh=10):
    result = np.zeros((size, size), dtype=np.uint8)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            dy, dx = int(img[i + 1, j]) - int(img[i, j]), int(img[i, j + 1]) - int(img[i, j])
            p_val = math.sqrt(dy ** 2 + dx ** 2)

            if p_val > thresh:
                result[i, j] = 255

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

def g2d(k, s = 1):
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

def g1d(k, s = 1):
    # Generates gaussian kernel based on the params
    kernel = np.zeros(k)
    radius = k // 2

    for i in range(k):
        x = i - radius
        kernel[i] = math.exp(-(x ** 2)/ (2 * s ** 2))

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def validate_img(img, s):
    return (img.shape[0] == s and img.shape[1] == s)


def main():
    if not len(sys.argv) == 4:
        raise ValueError(f"Invalid args\nUsage: python solution.py <img1 path> <img2 path> <img size>")

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    n = int(sys.argv[3])


    ##### TASK 1 - Image Filtering & Convolution #####
    m = 3
    f1 = g2d(m, 4)    # gaussian blur
    img1_blur = apply_filter(img1.copy(), m, n, f1)
    cv2.imwrite('img1_blur.png', img1_blur)

    # simple 3x3 sharpening kernel
    f2 = np.loadtxt(FILTER, dtype=np.float32)
    #f2 = normalize(f2)
    img1_sharpened = apply_filter(img1.copy(), m, n, f2)
    cv2.imwrite('img1_sharpened.png', img1_sharpened)


    ##### TASK 2 - Edge Detection #####
    m = 9
    for i in [.5, 1, 2, 3.0]:
        for j in [20, 30, 40, 50]:
            # denoise with edge detection
            f3 = g1d(9, i)
            pic1_smoothed = apply_separable_filter(img1.copy(), m, n, f3)

            # calculate edges
            pic1_edges = calculate_edges(pic1_smoothed, n, j)
            cv2.imwrite(f"pic1_edges_sigma{i}_thresh{j}.png", pic1_edges)




if __name__ == '__main__':
    main()