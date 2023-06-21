import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# 获取系统默认字体
default_font = plt.rcParams['font.sans-serif']

# 设置Matplotlib的字体为系统默认字体
plt.rcParams['font.sans-serif'] = default_font
plt.rcParams['font.sans-serif'] = ['SimHei']
# 导入数据
res = pd.read_excel('1.xlsx', sheet_name='Sheet2', header=None)
res = np.array(res)

# 划分训练集和测试集
temp = np.random.permutation(3943)
P_train = res[temp[:3154], :20]
T_train = res[temp[:3154], 20]
P_test = res[temp[3154:], :20]
T_test = res[temp[3154:], 20]

# 数据归一化
P_train_normalized = normalize(P_train)
P_test_normalized = normalize(P_test)

# 定义KNN模型
knn = KNeighborsClassifier()

# 定义参数网格
param_grid = {'n_neighbors': [3, 5, 7, 10],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

# 进行网格搜索
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(P_train_normalized, T_train)

# 获取交叉验证结果
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']
params = cv_results['params']

# 可视化交叉验证结果对比
plt.figure(figsize=(12, 6))
param_combinations = len(params)
x_ticks = np.arange(param_combinations)
bar_width = 0.2

for i, param in enumerate(params):
    n_neighbors = param['n_neighbors']
    weights = param['weights']
    metric = param['metric']
    label = f'n_neighbors: {n_neighbors}, weights: {weights}, metric: {metric}'
    scores = mean_scores[i::param_combinations]
    plt.bar(x_ticks + (i * bar_width), scores, bar_width, label=label)

plt.xlabel('参数组合')
plt.ylabel('平均测试得分')
plt.title('交叉验证结果对比')
plt.xticks(x_ticks + (bar_width * (param_combinations - 1) / 2), np.arange(param_combinations))
plt.legend()
plt.grid()

plt.show()
