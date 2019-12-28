import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_type = ['经向样本','块状疵点','纬向疵点','正常疵点']

train_x = []
train_y = []

file_path = "C:\\Users\\LAB\\Desktop\\带学生项目\\No4\\测试样本\\{}\\"

for i in image_type:
    sample_type = file_path.format(i)
    # print(sample_type)
    file_lists = os.listdir(sample_type)
    # print(list)
    for li in file_lists:
        img_path = sample_type+li
        # print(img_path)
        img = np.array(Image.open(img_path))
        v_img = np.vstack(img).ravel()

        train_y.append(i)
        train_x.append(v_img)

np.savetxt("x.txt",train_x,fmt="%s",newline='\n')
np.savetxt("y.txt",train_y,fmt="%s",newline='\n')