import os
import cv2
import numpy as np
import pandas as pd

# 设置rawdata文件夹的路径
folder_path = 'rawdata'

# 设置输出文件夹的路径
output_folder = 'fix'

# 如果输出文件夹不存在，则创建它
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 存储出现在打印语句中的数字集合
invalid_images = []

# 遍历每个文件
for file_name in file_list:
    # 创建完整的文件路径
    file_path = os.path.join(folder_path, file_name)

    # 读取图像
    image = np.fromfile(file_path, dtype=np.ubyte)
    image = image.reshape(128, -1)

    # 计算图像的像素平均值
    avg_pixel_value = np.mean(image)

    # 设置阈值来判断图像是否为黑色或无效图像
    threshold_black = 5
    threshold_white = 200

    # 如果平均像素值小于黑色阈值或大于白色阈值，将其视为无效图像，跳过保存步骤
    if avg_pixel_value < threshold_black or avg_pixel_value > threshold_white:
        print(f"无效图像文件: {file_name}")
        # 提取文件名中的数字
        digit = ''.join(filter(str.isdigit, file_name))
        invalid_images.append(int(digit))

        continue

    # 使用.jpg扩展名创建输出文件路径
    output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')

    # 将图像保存为JPG
    cv2.imwrite(output_path, image)

    # 打印进度
    # print(f"将 {file_name} 转换为JPG格式，并保存在{output_folder}文件夹中。")

print("转换完成！")

# 删除combined_data.xlsx中包含无效图像数字的行
data_path = 'combined_data.xlsx'
df = pd.read_excel(data_path)

# 获取第一列中包含无效图像数字的行的索引
invalid_rows = df[df.iloc[:, 0].isin(invalid_images)].index

# 删除这些行
df.drop(invalid_rows, inplace=True)

# 保存修改后的数据到combined_data.xlsx
df.to_excel(data_path, index=False)
