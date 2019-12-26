import cv2

img_rgb = cv2.imread('/Users/apple/Downloads/IMG_0298.JPG')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

img_edge = cv2.adaptiveThreshold(img_gray,
                                 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,
                                 blockSize=3,
                                 C=2)

cv2.imwrite('new_image.jpg',img_edge)

