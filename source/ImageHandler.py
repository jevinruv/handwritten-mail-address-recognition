import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess(img, imgSize):
    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([1, 1])

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (min(wt, int(w / f)), min(ht, int(h / f)))
    img = cv2.resize(img, newSize)

    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img

# def preprocess(img, imgSize):
#     img = cv2.resize(img, imgSize)
#
#     # increase contrast
#     pxmin = np.min(img)
#     pxmax = np.max(img)
#     imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
#
#     # increase line width
#     kernel = np.ones((3, 3), np.uint8)
#     imgMorph = cv2.erode(imgContrast, kernel, iterations=1)
#
#     return imgMorph

def pre():
    img = cv2.imread('in.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))

    # increase contrast
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

    # write
    cv2.imwrite('out.png', imgMorph)


def preprocess_mod():
    img = cv2.imread('in.png', cv2.IMREAD_GRAYSCALE)

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([1, 1])

    # create target image and copy sample image into it
    (wt, ht) = (128, 32)
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (min(wt, int(w / f)), min(ht, int(h / f)))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    plt.imshow(img)
    plt.show()


# pre()
# preprocess_mod()


# img = cv2.imread('../resources/test1.png')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
#
img1 = cv2.imread('../resources/test1.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img1, cmap='gray')
plt.show()

