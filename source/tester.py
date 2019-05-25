import cv2
import numpy as np

image = cv2.imread('../resources/test0.png')


def address_to_lines(img_address):
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


def line_to_words(img_line):
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


def split_text(image):
    word_list = []

    lines = address_to_lines(image)

    for line in lines:
        words = line_to_words(line)
        word_list.extend(words)

    return word_list


words = split_text(image)

for img in words:
    cv2.imshow("word", img)
    cv2.waitKey(0)
