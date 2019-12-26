import cv2
import numpy as np
'''
作用：gamma矫正通常用于电视和监视器系统中重现摄像机拍摄的画面．
在图像处理中也可用于调节图像的对比度，减少图像的光照不均和局部阴影．
原理： 通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，
即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正
'''


# img = cv2.imread('/Users/apple/Desktop/Bolt.jpeg')
# img1 = np.power(img/float(np.max(img)),1/1.5)
# img2 = np.power(img/float(np.max(img)),1.5)
#
# cv2.imshow('src',img)
# cv2.imshow('gamma=1/1.5',img1)
# cv2.imshow('gamma=1.5',img2)
# cv2.waitKey(0)

# im = cv2.imread('/Users/apple/Desktop/Bolt.jpeg')
# im = np.float32(im) / 255.0

# img = cv2.GaussianBlur(im,(3,3),0)

# gx = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=1)
# gy = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=1)
#
# mag,angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)
#
# cv2.imshow('1',gx)
# cv2.imshow('2',gy)
# cv2.imshow('3',mag)
# cv2.imshow('4',angle)
# cv2.waitKey(0)

# import matplotlib.pyplot as plt
# from skimage.feature import hog
# from skimage import data, exposure
# image = cv2.imread('/Users/apple/Desktop/Bolt.jpeg')
# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.datasets import load_digits

digits = load_digits()

image = digits.images[0]

fd,hog_image = hog(image,orientations=8,pixels_per_cell=(16,16),
                   cells_per_block=(1,1),visualize=True)
fig,(ax1,ax2) = plt.subplots(1,2, figsize=(8,4),sharex=True,sharey=True)
ax1.axis('off')
ax1.imshow(image,cmap=plt.cm.gray)
ax1.set_title('input image')
hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled,cmap=plt.cm.gray)
ax2.set_title('hog')
plt.show()
