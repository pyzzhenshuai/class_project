import os
import cv2
import numpy as np
from skimage.feature import hog
import pandas as pd
from sklearn.decomposition import PCA

# 设置文件夹路径
folder = 'fix'
# 获取文件夹中所有jpg文件
jpegFiles = [f for f in os.listdir(folder) if f.endswith('.jpg')]

# 创建一个存储特征值的列表
features = []

# 遍历每个文件
for filename in jpegFiles:
    # 读取图像
    img = cv2.imread(os.path.join(folder, filename))

    # 将图像调整为指定大小
    img = cv2.resize(img, (128, 128))

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用HOG方法提取特征
    hogFeature = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # 将特征值添加到列表中
    features.append(hogFeature)

# 将特征值转换为NumPy数组
features = np.array(features)

# 创建PCA对象，将特征维度降低到n_components
n_components = 20
pca = PCA(n_components=n_components)

# 对特征进行降维
features_reduced = pca.fit_transform(features)

# 创建DataFrame对象
df = pd.DataFrame(features_reduced)

# 保存特征值到Excel文件
df.to_excel('特征值.xlsx', index=False)
