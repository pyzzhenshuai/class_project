import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 获取系统默认字体
default_font = fm.findfont(fm.FontProperties())

# 设置Matplotlib的字体为系统默认字体
plt.rcParams['font.sans-serif'] = [default_font]
plt.rcParams['font.sans-serif']=['SimHei']

# 清空环境变量
# warning off             # 在Python中不需要关闭报警信息
plt.close('all')            # 关闭开启的图窗
np.random.seed(42)          # 设置随机种子

# 导入数据
data = pd.read_excel('1.xlsx', sheet_name='Sheet2',header=None)
res = data.values

# 划分训练集和测试集
np.random.shuffle(res)      # 打乱数据
P_train = res[:3548, :20]
T_train = res[:3548, 20]
P_test = res[3548:, :20]
T_test = res[3548:, 20]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
p_train = scaler.fit_transform(P_train)
p_test = scaler.transform(P_test)
t_train = T_train
t_test = T_test

# 创建模型
c = 4       # 惩罚因子
g = 1.7  # 径向基函数参数
model = svm.SVC(kernel='rbf', C=c, gamma=g)
model.fit(p_train, t_train)

# 仿真测试
t_sim1 = model.predict(p_train)
t_sim2 = model.predict(p_test)

# 性能评价
error1 = accuracy_score(t_train, t_sim1) * 100
error2 = accuracy_score(t_test, t_sim2) * 100

# 数据排序
index_1 = np.argsort(t_train)
index_2 = np.argsort(t_test)
t_train_sorted = t_train[index_1]
t_test_sorted = t_test[index_2]
t_sim1_sorted = t_sim1[index_1]
t_sim2_sorted = t_sim2[index_2]

# 绘图
plt.figure()
plt.plot(range(len(t_train)), t_train_sorted, 'r-*', label='真实值')
plt.plot(range(len(t_train)), t_sim1_sorted, 'b-o', label='预测值', linewidth=1)
plt.legend()
plt.xlabel('预测样本')
plt.ylabel('预测结果')
string = '训练集预测结果对比\n准确率 = {:.2f}%'.format(error1)
plt.title(string)
plt.grid()

plt.figure()
plt.plot(range(len(t_test)), t_test_sorted, 'r-*', label='真实值')
plt.plot(range(len(t_test)), t_sim2_sorted, 'b-o', label='预测值', linewidth=1)
plt.legend()
plt.xlabel('预测样本')
plt.ylabel('预测结果')
string = '测试集预测结果对比\n准确率 = {:.2f}%'.format(error2)
plt.title(string)
plt.grid()

# 混淆矩阵
plt.figure()
cm_train = confusion_matrix(t_train, t_sim1)
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_train, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix for Train Data')
plt.colorbar()
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.grid(False)

plt.figure()
cm_test = confusion_matrix(t_test, t_sim2)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_test, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.grid(False)

plt.show()
