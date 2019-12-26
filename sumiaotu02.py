import cv2

img_rgb = cv2.imread('/Users/apple/Downloads/IMG_0298.JPG')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

img_blur = cv2.GaussianBlur(img_gray,
                            ksize=(21,21),
                            sigmaX=0,
                            sigmaY=0)

img_new = cv2.divide(img_gray,img_blur,scale=255)

cv2.imwrite('new_image02.jpg',img_new)