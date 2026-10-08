import cv2
import numpy as np
import math

IMG_1 = 'pic1grey300.jpg'
IMG_2 = 'pic2grey300.jpg'
FILTER = 'sharpening_filter3x3.txt'

def apply_filter(img, img_filter):
    # This function convolves images using the specified filter

    m = len(img_filter)
    n = img.shape[0]

    for y in range(m//2, n - m//2):
        for x in range(m//2, n - m//2):
            p_val = 0

            if not check_conv_boundary(y, x, m, n):
                p_val = convolve(img, y, x, img_filter, m)

            img[y, x] = p_val

    return img
            

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
        y >= n - m or
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


def main():
    img1 = cv2.imread(IMG_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMG_2, cv2.IMREAD_GRAYSCALE)

    ### TASK 1 ###
    f1 = g(7, 4)    # gaussian blur
    img1_blur = apply_filter(img1.copy(), f1)
    cv2.imwrite('img1_blur.png', img1_blur)

    f2 = np.loadtxt(FILTER, dtype=float)    # simple 3x3 sharpening kernel
    f2 = normalize(f2)
    img1_sharpened = apply_filter(img1.copy(), f2)
    cv2.imwrite('img1_sharpened.png', img1_sharpened)


if __name__ == '__main__':
    main()