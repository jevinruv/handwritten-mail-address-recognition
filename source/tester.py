import cv2
import numpy as np
from ImageHandler import ImageHandler

image = cv2.imread('../resources/qw.png')
image = cv2.imread('../resources/test0.png')

# imageHandler = ImageHandler()
#
# words = imageHandler.split_text(image)
#
# for img in words:
#     cv2.imshow("word", img)
#     cv2.waitKey(0)


# img = cv2.imread('../resources/IMG_20190707_085650.jpg', cv2.IMREAD_UNCHANGED)
#
# print('Original Dimensions : ', img.shape)
#
# width = 500
# height = 300
# dim = (width, height)
#
# # resize image
# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
# print('Resized Dimensions : ', resized.shape)
#
# cv2.imshow("Resized image", resized)
#
# cv2.imwrite("edited.png", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('orig', image)
cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('second', thresh)
cv2.waitKey(0)

# dilation
kernel = np.ones((5, 7), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

# find contours
im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y + h, x:x + w]

    # show ROI
    cv2.imshow('segment no:' + str(i), roi)
    # cv2.imwrite("segment_no_" + str(i) + ".png", roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    # cv2.waitKey(0)

# cv2.imwrite('final_bounded_box_image.png', image)
cv2.imshow('marked areas', image)
cv2.waitKey(0)
