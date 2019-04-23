import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageHandler:

    def preprocess(self, img, imgSize):
        # transform damaged files to black images
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

    def preprocess_normal_handwriting(self, img):
        # increase contrast
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

        # increase line width
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

        return imgMorph

    def preprocess_mod(self, img, imgSize):
        # transform damaged files to black images
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

    def split_text(self, img, split_type):
        text_list = []

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # binarize
        ret, thresh = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)

        if (split_type == 'word'):
            kernel = np.ones((5, 10), np.uint8)
        else:
            kernel = np.ones((5, 100), np.uint8)

        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        _, contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, contour in enumerate(sorted_contours):
            # apply bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # retrieve text line & add to list
            line = img[y:y + h, x:x + w]
            text_list.append(line)

            # show text line
            # cv2.imshow('line no:' + str(i), line)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
            # cv2.waitKey(0)

        # cv2.imshow('marked areas', image)
        # cv2.waitKey(0)
        return text_list

# file_test_img = '../resources/test1.png'

# img = cv2.imread(file_test_img)
# result = preprocess_normal_handwriting(img)
# cv2.imwrite('out.png', result)

# img = cv2.imread('../resources/test1.png')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# img1 = cv2.imread('../resources/test1.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(img1, cmap='gray')
# plt.show()
