import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

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

# 获取最佳参数和模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 使用最佳模型进行预测
T_sim1 = best_model.predict(P_train_normalized)
T_sim2 = best_model.predict(P_test_normalized)

# 计算准确率
error1 = accuracy_score(T_train, T_sim1) * 100
error2 = accuracy_score(T_test, T_sim2) * 100

# 绘图
plt.figure()
plt.plot(np.arange(len(T_train)), T_train, 'r-*', np.arange(len(T_train)), T_sim1, 'b-o', linewidth=1)
plt.legend(['真实值', '预测值'])
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('训练集预测结果对比\n准确率={:.2f}%'.format(error1))
plt.grid()

plt.figure()
plt.plot(np.arange(len(T_test)), T_test, 'r-*', np.arange(len(T_test)), T_sim2, 'b-o', linewidth=1)
plt.legend(['真实值', '预测值'])
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('测试集预测结果对比\n准确率={:.2f}%'.format(error2))
plt.grid()

# 混淆矩阵
cm_train = confusion_matrix(T_train, T_sim1)
cm_test = confusion_matrix(T_test, T_sim2)

disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=np.unique(T_train))
disp_train.plot()
plt.title('Confusion Matrix for Train Data')

disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(T_test))
disp_test.plot()
plt.title('Confusion Matrix for Test Data')

plt.show()

# 打印最佳参数
print("最佳参数：", best_params)
