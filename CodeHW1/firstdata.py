import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_excel('data/firstdata.xlsx')
a = data[['TV', 'Radio', 'Newspaper']]
# A矩阵
A = np.matrix(a)
AT = A.T
# 系数b
b = np.matrix(data['Sales'])
# 最小二乘问题的解析解
x = ((AT * A).I) * AT * (b.T)
pre = A * x
p = plt.figure(figsize=(14, 14))
ax1 = p.add_subplot(2, 2, 1)
label = [i for i in range(0, 200)]
# 将y排序
y = data['Sales'].sort_values()
pre = pd.DataFrame(pre)
# 将求出系数带入的预测值排序
pre = pre[0].sort_values()
# print(pre-y)
# 求残差（其实不太有必要只是显的好看）
e = (y - pre).sort_values()
plt.ylabel('residual error')
plt.title('using the analytical solution')
plt.scatter(label, e, marker='o')
# 找到最小范数和最小值的解
print('解析解求得的系数：\n', x)
print('解析解求得的范数最小值:\n', 1 / 2 * np.linalg.norm(e, ord=2))

# 固定步长
# 设定固定步长a1
a1 = 0.00000001
# 设定初始迭代点(
xf = np.matrix([[0.1], [0.25], [0.02]])
x1 = xf
ax2 = p.add_subplot(2, 2, 2)
pref = pd.DataFrame(A * x1 - b.T)
pref = pref[0].sort_values()
x1 = xf
# 求了一下不同步数的残差（其实没太有必要）
for i in range(200):
    x1 = x1 - a1 * AT * (A * x1 - b.T)
    if i % 100 == 0:
        preg = pd.DataFrame(A * x1 - b.T)
        preg = preg[0].sort_values()
        plt.scatter(label, preg, color='r')
    if i == 199:
        preg = pd.DataFrame(A * x1 - b.T)
        preg = preg[0].sort_values()
        plt.scatter(label, preg, color='y')
plt.scatter(label, e, color='b')
plt.ylabel('residual error')
plt.title('using the fixed step')
plt.legend(
    ['original condition residual error(i=0)', 'when i=100', 'when i=199', "analytical solution's residual error"])

ax3 = p.add_subplot(2, 2, 3)
x1 = xf
result = []
minnorm = 1 / 2 * np.linalg.norm(pref, ord=2)
# 求了一下不同步数得到的二范数的值，可以看出来后面相差不太大，但是这个最小值的得出很依赖于选的步长和初始点
for j in range(100):
    x1 = x1 - a1 * AT * (A * x1 - b.T)
    preg = pd.DataFrame(A * x1 - b.T)
    norm = 1 / 2 * np.linalg.norm(preg, ord=2)
    if norm < minnorm:
        mini = j
        minnorm = norm
        minx = x1
    result.append(norm)
label2 = [i for i in range(100)]
# 找到这种初始情况和步长的范数最小值
print('固定步长求得的范数最小值:\n', minnorm)
# 范数最小时的系数
print('范数最小时的系数：\n', minx)
# 解析解求得的系数
print('解析解求的系数：\n', x)
plt.scatter(label2, result)
plt.ylabel('norm2')
plt.show()

# 后退线性搜索
# 设定参数ba,bb
xf2 = np.matrix([[0.06], [0.25], [0.02]])
# ba是α，bb是β
ba = 0.1
bb = 0.3
t = 0.000001
x2 = xf2
# print(ba*t*(AT*(A*x2-b.T)).T*(AT*(A*x2-b.T)))
cnt = 0
result2 = []
# 直接将书上的条件抄上去，但不太懂为啥，同样求不同步数求得的范数值，同样很依赖于初始点和t的选择，还有α和β的
while 1 / 2 * np.linalg.norm(A * (x2 - t * AT * (A * x2 - b.T)) - b.T, ord=2) > 1 / 2 * np.linalg.norm(A * x2 - b.T,
                                                                                                       ord=2) - ba * t * (
        AT * (A * x2 - b.T)).T * (AT * (A * x2 - b.T)):
    t = bb * t
    cnt += 1
    x2 -= t * AT * (A * x2 - b.T)
    result2.append(1 / 2 * np.linalg.norm(A * x2 - b.T, ord=2))
print('后退搜索法求出的解：\n', x2)
print('后退搜索法求出的范数最小值：\n', 1 / 2 * np.linalg.norm(A * x2 - b.T, ord=2))
ax4 = p.add_subplot(2, 2, 4)
label3 = [i for i in range(cnt)]
plt.scatter(label3, result2, color='b')
plt.ylabel('norm2')
plt.show()
