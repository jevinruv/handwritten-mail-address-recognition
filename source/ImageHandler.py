import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageHandler:

    def preprocess(self, img, img_size):
        # transform damaged files to black images
        if img is None:
            img = np.zeros([1, 1])

        # create target image and copy sample image into it
        (wt, ht) = img_size
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

    def address_to_lines(self, img_address):
        # grayscale
        gray = cv2.cvtColor(img_address, cv2.COLOR_BGR2GRAY)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((5, 50), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        # find contours
        im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        sorted_ctrs = sorted(ctrs,
                             key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img_address.shape[1])

        lines = []

        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = img_address[y:y + h, x:x + w]

            lines.append(roi)

        return lines

    def line_to_words(self, img_line):
        # grayscale
        gray = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((5, 15), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        # find contours
        im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        words = []

        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = img_line[y:y + h, x:x + w]
            words.append(roi)

        return words

    def split_text(self, image):
        word_list = []

        img = cv2.imread(image)
        lines = self.address_to_lines(img)

        for line in lines:
            words = self.line_to_words(line)
            word_list.extend(words)

        return word_list

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
