import cv2
import numpy as np
from ImageHandler import ImageHandler

image = cv2.imread('../resources/qw.png')
image = cv2.imread('../resources/test3.png')

imageHandler = ImageHandler()

words = imageHandler.split_text(image)

for img in words:
    cv2.imshow("word", img)
    cv2.waitKey(0)

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
