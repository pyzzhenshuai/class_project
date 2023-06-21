import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

plt.rcParams['font.family'] = 'SimHei'  # 选择合适的字体
plt.rcParams['axes.unicode_minus'] = False  # 禁用减号显示为方块的警告

# 导入数据
data = pd.read_excel('1.xlsx', sheet_name='Sheet2')
np.random.seed(0)
temp = np.random.permutation(len(data))

train_indices = temp[:3548]
test_indices = temp[3548:]

P_train = data.iloc[train_indices, :20].values.T
T_train = data.iloc[train_indices, 20].values.reshape(3548, 1)
M = P_train.shape[1]

P_test = data.iloc[test_indices, :20].values.T
T_test = data.iloc[test_indices, 20].values.reshape(len(test_indices), 1)
N = P_test.shape[1]

# 数据归一化
scaler = MinMaxScaler()
p_train = scaler.fit_transform(P_train.T).T
p_test = scaler.transform(P_test.T).T
t_train = T_train
t_test = T_test

# Parameter Tuning
param_grid = {
    'n_estimators': [1, 2, 4, 10, 20, 50, 75, 100, 150, 200, 500, 1000],
    'min_samples_leaf': [5, 10, 15, 20],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                           param_grid=param_grid,
                           cv=5)

grid_search.fit(p_train.T, t_train.ravel())

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Create the random forest with the best parameters
rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                            min_samples_leaf=best_params['min_samples_leaf'],
                            max_features=best_params['max_features'],
                            random_state=0)

rf.fit(p_train.T, t_train.ravel())

# 仿真测试
t_sim1 = rf.predict(p_train.T)
t_sim2 = rf.predict(p_test.T)

# 性能评价
error1 = np.sum(t_sim1 == t_train.ravel()) / M * 100
error2 = np.sum(t_sim2 == t_test.ravel()) / N * 100

# 绘制误差曲线
grid_results = grid_search.cv_results_
mean_scores = grid_results['mean_test_score']
params = grid_results['params']
n_estimators = len(params)

plt.figure()
plt.plot(range(1, n_estimators + 1), 1 - mean_scores, 'b-', linewidth=1)
plt.legend(['误差曲线'])
plt.xlabel('决策树数目')
plt.ylabel('误差')
plt.xlim(1, n_estimators)
plt.grid()

# 绘制特征重要性
importance = rf.feature_importances_
plt.figure()
plt.bar(range(P_train.shape[0]), importance)
plt.legend(['重要性'])
plt.xlabel('特征')
plt.ylabel('重要性')

# 数据排序
index_1 = np.argsort(t_train.ravel())
index_2 = np.argsort(t_test.ravel())
T_train_sorted = np.sort(t_train.ravel())
T_test_sorted = np.sort(t_test.ravel())
T_sim1_sorted = t_sim1[index_1]
T_sim2_sorted = t_sim2[index_2]

# 绘图
plt.figure()
plt.plot(range(M), T_train_sorted, 'r-*', range(M), T_sim1_sorted, 'b-o', linewidth=1)
plt.legend(['真实值', '预测值'])
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('训练集预测结果对比\n准确率 = {:.2f}%'.format(error1))
plt.grid()

plt.figure()
plt.plot(range(N), T_test_sorted, 'r-*', range(N), T_sim2_sorted, 'b-o', linewidth=1)
plt.legend(['真实值', '预测值'])
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('测试集预测结果对比\n准确率 = {:.2f}%'.format(error2))
plt.grid()

# 混淆矩阵
cm_train = confusion_matrix(t_train.ravel(), t_sim1)
cm_test = confusion_matrix(t_test.ravel(), t_sim2)

plt.figure()
plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Train Data')
plt.colorbar()
for i in range(len(cm_train)):
    for j in range(len(cm_train)):
        plt.text(j, i, str(cm_train[i][j]), ha='center', va='center', color='red')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

plt.figure()
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
for i in range(len(cm_test)):
    for j in range(len(cm_test)):
        plt.text(j, i, str(cm_test[i][j]), ha='center', va='center', color='red')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

# Classification Report
report_train = classification_report(t_train.ravel(), t_sim1)
report_test = classification_report(t_test.ravel(), t_sim2)

print("Classification Report (Train):\n", report_train)
print("Classification Report (Test):\n", report_test)
