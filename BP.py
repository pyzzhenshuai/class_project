import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # 使用StandardScaler进行数据标准化
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 获取系统默认字体
default_font = fm.findfont(fm.FontProperties())

# 设置Matplotlib的字体为系统默认字体
plt.rcParams['font.sans-serif'] = [default_font]
plt.rcParams['font.sans-serif'] = ['SimHei']

# 导入数据
data = pd.read_excel('1.xlsx', sheet_name='Sheet2')
res = data.values

# 随机排列数据的索引
np.random.seed(0)
temp = np.random.permutation(3943)

# 根据打印输出，确保切片操作不会超出数组的大小范围
print("res shape:", res.shape)

# 划分训练集和测试集
P_train = res[temp[:3542] - 1, :20].T
T_train = res[temp[:3542] - 1, 20].reshape(-1, 1).T

M = P_train.shape[1]

P_test = res[temp[3548:]][:, :20].T
T_test = res[temp[3548:], 20].reshape(-1, 1).T
N = P_test.shape[1]

# 将 T_train 中的所有 0 替换为 1
T_train[T_train == 0] = 2
T_test[T_test == 0] = 2

# 数据标准化
scaler = StandardScaler()  # 使用StandardScaler进行数据标准化
p_train = scaler.fit_transform(P_train.T).T
p_test = scaler.transform(P_test.T).T
t_train = np.eye(6)[T_train.flatten().astype(int) - 1].T
t_test = np.eye(6)[T_test.flatten().astype(int) - 1].T

# 建立模型
net = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, learning_rate_init=0.001, random_state=0)
# 调整了隐藏层大小为两个50节点的隐藏层，学习率为0.001

# 训练模型
net.fit(p_train.T, t_train.T)

# 仿真测试
t_sim1 = net.predict(p_train.T)
t_sim2 = net.predict(p_test.T)

# 数据反归一化
T_sim1 = np.argmax(t_sim1, axis=1) + 1
T_sim2 = np.argmax(t_sim2, axis=1) + 1

# 性能评价
error1 = np.sum(T_sim1 == T_train.flatten()) / M * 100
error2 = np.sum(T_sim2 == T_test.flatten()) / N * 100

# 绘图
plt.figure()
plt.plot(range(1, M + 1), T_train.flatten(), 'r-*', label='真实值')
plt.plot(range(1, M + 1), T_sim1, 'b-o', label='预测值')
plt.legend()
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('训练集预测结果对比\n准确率={:.2f}%'.format(error1))
plt.grid(True)
plt.xlim(1, M)

plt.figure()
plt.plot(range(1, N + 1), T_test.flatten(), 'r-*', label='真实值')
plt.plot(range(1, N + 1), T_sim2, 'b-o', label='预测值')
plt.legend()
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('测试集预测结果对比\n准确率={:.2f}%'.format(error2))
plt.grid(True)
plt.xlim(1, N)

# 混淆矩阵
cm_train = confusion_matrix(T_train.flatten(), T_sim1)
cm_test = confusion_matrix(T_test.flatten(), T_sim2)

# Plotting Confusion Matrix for Train Data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cm_train, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix for Train Data')
plt.colorbar()
plt.xlabel('预测类别')
plt.ylabel('真实类别')

# Add numerical values as annotations with darker gray color
for i in range(cm_train.shape[0]):
    for j in range(cm_train.shape[1]):
        plt.text(j, i, str(cm_train[i, j]), ha='center', va='center', color='darkgray')

# Plotting Confusion Matrix for Test Data
plt.subplot(1, 2, 2)
plt.imshow(cm_test, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.xlabel('预测类别')
plt.ylabel('真实类别')

# Add numerical values as annotations with darker gray color
for i in range(cm_test.shape[0]):
    for j in range(cm_test.shape[1]):
        plt.text(j, i, str(cm_test[i, j]), ha='center', va='center', color='darkgray')

plt.tight_layout()
plt.show()
