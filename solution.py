""" 
Project 2: Edge Detection
-----------------------------------------------------------------------


Usage: 
    python solution.py <img1 path> <img2 path> <img size>

Example:
    python solution.py pic1grey300.jpg pic2grey300.jpg 300

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
    # Canny edge detection with two threshhold

    gradents = get_gradients(img, size)
    result = np.zeros((size, size), dtype=np.uint8)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            dy, dx = gradents[i, j][0], gradents[i, j][1]
            p_val = math.sqrt(dy ** 2 + dx ** 2)

            if p_val > thresh:
                result[i, j] = 255

    return result

def calculate_corners(img, size, thresh=40):
    gradents = get_gradients(img, size)
    mt = np.zeros((size, size, 3), dtype=np.float32)

    # 2nd moment vals
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            dy, dx = gradents[i, j][0], gradents[i, j][1]   # works better without the arbitary division
            second_moment_vals = np.array([dx ** 2, dy ** 2, dy * dx])
            mt[i, j] = second_moment_vals

    result = np.zeros((size, size), dtype=np.float64)

    # smooth using gaussian
    f = g1d(11, 5.5)
    r = 5
    mtA, mtB, mtC = mt[:, :, 0], mt[:, :, 1], mt[:, :, 2]
    mtA = apply_separable_filter(mtA, 11, size, f)
    mtB = apply_separable_filter(mtB, 11, size, f)
    mtC = apply_separable_filter(mtC, 11, size, f)


    # detect corners using threshold
    for i in range(r, size - r):
        for j in range(r, size - r):

            R_val = (mtA[i, j] * mtB[i, j] - mtC[i, j] ** 2) - .04 * (mtA[i, j] + mtB[i, j]) ** 2

            if R_val > thresh:
                result[i, j] = R_val

    result2 = np.zeros((size, size), dtype=np.uint8)
    
    # non maxima suppression
    for i in range(2 * r, size - 2 * r):    # thick border to eliminate edge points
        for j in range(2 * r, size - 2 * r):
            if local_maxima(result, i, j):
                result2[i, j] = 255

    return result2, gradents

def local_feature_descriptor(corners, gradients, img_size, sample_size):
    # generates grad vectors describing features around each corner point

    result = []

    for i in range(img_size):
        for j in range(img_size):
            if corners[i, j] == 255:
                hist = get_grad_histogram(gradients, i, j, sample_size)
                print((i, j, hist))
                result.append((i, j, hist))

    return result

def match_descriptors(hist1, hist2, size):
    for i in range(0, size, 30):
        for j in range(0, size, 30):

def get_gradients(img, size):
    h = img.astype(np.float64)

    result = np.zeros((size, size, 2), dtype=np.float64)
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            result[i, j] = np.array([h[i + 1, j] - h[i, j], h[i, j + 1] - h[i, j]])

    return result


def get_grad_histogram(gradients, y, x, size, normalize=True):
    radius = size // 2
    hist = np.zeros(8, dtype=np.float64)

    for i in range(-1 * radius, radius + 1):
        for j in range(-1 * radius, radius + 1):
            dy, dx = gradients[y + i, x + j][0], gradients[y + i, x + j][1]
            mag =  math.sqrt(dy ** 2 + dx ** 2)
            deg = math.degrees(math.atan2(dy, dx)) % 360

            hist[int(deg / 45)] += mag

    if normalize:
        # normalize hist
        indx = np.argmax(hist)
        result = np.zeros(8, dtype=np.float64)

        for i in [4, 5, 6, 7, 0, 1, 2, 3]:
            result[i] = hist[indx]
            indx += 1
            if indx > 7:
                indx = 0
    else:
        result = hist

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

def local_maxima(img, x, y):
    # Is pixel x,y is the greatest!
    return (
        img[x, y] > img[x + 1, y] and
        img[x, y] > img[x - 1, y] and
        img[x, y] > img[x, y + 1] and
        img[x, y] > img[x, y - 1] and 
        img[x, y] > img[x - 1, y - 1] and 
        img[x, y] > img[x + 1, y + 1] and 
        img[x, y] > img[x - 1, y + 1] and 
        img[x, y] > img[x + 1, y - 1]
    )

def find_neighbors(img, x, y, thresh):
    # check if any nearby neighbors meet the threshold
    return (
        img[x, y] > img[x + 1, y] or
        img[x, y] > img[x - 1, y] or
        img[x, y] > img[x, y + 1] or
        img[x, y] > img[x, y - 1] or 
        img[x, y] > img[x - 1, y - 1] or 
        img[x, y] > img[x + 1, y + 1] or 
        img[x, y] > img[x - 1, y + 1] or 
        img[x, y] > img[x + 1, y - 1]
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

def clip (num, min, max):
    if num < min:
        return min

    if num > max:
        return max

    return num


def main():
    if not len(sys.argv) == 4:
        raise ValueError(f"Invalid args\nUsage: python solution.py <img1 path> <img2 path> <img size>")

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    n = int(sys.argv[3])


    print('##### TASK 1 - Image Filtering & Convolution #####')
    m = 3
    f1 = g2d(m, 4)    # gaussian blur
    img1_blur = apply_filter(img1, m, n, f1)
    cv2.imwrite('img1_blur.png', img1_blur)
    print(f'Success: img1_blur.png')

    # simple 3x3 sharpening kernel
    f2 = np.loadtxt(FILTER, dtype=np.float32)
    #f2 = normalize(f2)
    img1_sharpened = apply_filter(img1, m, n, f2)
    cv2.imwrite('img1_sharpened.png', img1_sharpened)

    print(f'Success: img1_sharpened.png')


    print('##### TASK 2 - Edge Detection #####')
    m = 9
    for i in [.5, 1, 2, 3.0]:
        for j in [20, 30, 40, 50]:
            # denoise with edge detection
            f3 = g1d(9, i)
            pic1_smoothed = apply_separable_filter(img1, m, n, f3)

            # calculate edges
            pic1_edges = calculate_edges(pic1_smoothed, n, j)
            cv2.imwrite(f"pic1_edges_sigma{i}_thresh{j}.png", pic1_edges)
            print(f'Success: pic1_edges_sigma{i}_thresh{j}.png')


    print('##### TASK 3 - Corner Detection #####')
    # Part a
    m = 9
    f4 = g1d(m, 2)
    pic1_smoothed_2 = apply_separable_filter(img1, m, n, f4)

    for thresh in range(0, 601, 150):
        corners_output = img1_color.copy()
        corners, gradients = calculate_corners(pic1_smoothed_2, 300, thresh=thresh)

        for i in range(n):
            for j in range(n):
                if corners[i, j] == 255:
                    corners_output[i, j] = [0, 0, 255]

        cv2.imwrite(f"harris_corner_detect_thresh{thresh}.png", corners_output)
        print(f'Success: harris_corner_detect_thresh{thresh}.png')

    # Part b
    print(f'local feature descriptors for img1: ')
    corners, gradients = calculate_corners(pic1_smoothed_2, 300, thresh=300)
    hists = local_feature_descriptor(corners, gradients, n, sample_size=9)


    print('##### TASK 4 - Image matching #####')
    pic2_smoothed = apply_separable_filter(img2, m, n, f4)
    corners2, gradients2 = calculate_corners(pic2_smoothed, 300, thresh=300)
    hist2 = local_feature_descriptor(corners2, gradients2, n, sample_size=9)


    
    

    




if __name__ == '__main__':
    main()