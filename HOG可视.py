import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# 读取原图像
img = cv2.imread('fix/1224.jpg')

# 将图像调整为指定大小
img_resized = cv2.resize(img, (128, 128))

# 转换为灰度图像
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 使用HOG方法提取特征
hogFeature, hogImage = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# 显示原图像与灰度处理后的图像对比
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale Image')
axes[1].axis('off')
plt.show()

# 显示HOG特征图像
plt.imshow(hogImage, cmap='gray')
plt.axis('off')
plt.title('HOG Visualization')
plt.show()
