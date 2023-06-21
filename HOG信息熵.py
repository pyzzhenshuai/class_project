import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import entropy
import matplotlib.pyplot as plt

# 读取原图像
img = cv2.imread('fix/1224.jpg')

# 将图像调整为指定大小
img_resized = cv2.resize(img, (128, 128))

# 转换为灰度图像
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 参数设置
orientations_list = [4, 8, 9, 12]  # orientations 参数取值列表
pixels_per_cell_list = [(4, 4), (8, 8), (16, 16)]  # pixels_per_cell 参数取值列表
cells_per_block_list = [(1, 1), (2, 2), (2, 4)]  # cells_per_block 参数取值列表

# 存储不同参数设置下的HOG特征的信息熵
hog_entropy_per_param = []

# 存储每个参数设置的字符串表示
param_strings = []

# 提取不同参数设置下的HOG特征
for orientations in orientations_list:
    for pixels_per_cell in pixels_per_cell_list:
        for cells_per_block in cells_per_block_list:
            # 使用HOG方法提取特征
            hog_feature = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

            # 计算特征的信息熵
            hog_entropy = entropy(hog_feature)
            hog_entropy_per_param.append(hog_entropy)

            # 生成参数设置的字符串表示
            param_string = f"orientations={orientations}, pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}"
            param_strings.append(param_string)

            # 打印当前参数设置和信息熵
            print(param_string)
            print(f"Entropy: {hog_entropy}")

# 绘制不同参数设置下的HOG特征的信息熵对比图
x_labels = [f"Param {i+1}" for i in range(len(hog_entropy_per_param))]
x = np.arange(len(hog_entropy_per_param))
y = hog_entropy_per_param

plt.bar(x, y)
plt.xlabel('Parameter Setting')
plt.ylabel('Entropy')
plt.xticks(x, x_labels, rotation='vertical')
plt.title('Entropy of HOG Features for Different Parameter Settings')
plt.tight_layout()
plt.show()

# 打印参数对应的字符串表示
for i, param_string in enumerate(param_strings):
    print(f"Param {i+1}: {param_string}")
