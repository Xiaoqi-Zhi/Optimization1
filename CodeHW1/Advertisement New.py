import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
data=pd.read_excel('data/firstdata.xlsx')
#对数据进行离差标准化
#定义一个函数
def minmaxscale(data):
    data=(data-data.min())/(data.max()-data.min())
    return data
a1=minmaxscale(data['TV'])
a2=minmaxscale(data['Radio'])
a3=minmaxscale(data['Newspaper'])
#求A矩阵
a1=np.matrix(a1)
a2=np.matrix(a2)
a3=np.matrix(a3)
A=np.concatenate((a1.T,a2.T,a3.T), axis=1)
AT=A.T
#求系数b
b=minmaxscale(np.matrix(data['Sales']))


#方法1：最小二乘问题的解析解
xa=((AT*A).I)*AT*(b.T)
#求出预测值
pre=A*xa
print('解析解求得的系数：\n',xa)
print('解析解求得的范数最小值:\n',1/2*np.linalg.norm(A*xa-b.T,ord=2))

print('----------------------------------------')
#方法2：固定步长
#设定固定步长a1(切换不同步长真的很重要)
a1=0.01
#设定初始迭代点
xf=np.matrix([[10],[10],[10]])
#设置画布
p=plt.figure(figsize=(14,14))
ax1=p.add_subplot(2,2,1)
x1=xf
result=[]#用于存放得到的结果
#开始进行迭代
start=time.time_ns()
for j in range(1000):
    x1=x1-a1*AT*(A*x1-b.T)
    preg=pd.DataFrame(A*x1-b.T)
    norm=1/2*np.linalg.norm(preg,ord=2)
    if 1/2*np.linalg.norm(A*x1-b.T,ord=2)-1/2*np.linalg.norm(A*(x1-a1*AT*(A*x1-b.T))-b.T)<0.00001:#如果前后两值相差小于0.00001则达到停止条件
        break
    result.append(norm)
#设置画布横坐标
label=[i for i in range(j)]
print('使用固定步长法的时间为：\n',time.time_ns()-start)
print('使用固定步长法的迭代次数为：\n',j)
print('使用固定步长法求得的范数最小时的系数（解）为：\n',x1)
print('使用固定步长法求得的范数最小值：\n',1/2*np.linalg.norm(A*x1-b.T,ord=2))
#作图
plt.scatter(label,result,color='b')
x=np.linspace(0,100,100)
plt.axhline(1/2*np.linalg.norm(A*xa-b.T,ord=2),color='r')
plt.ylabel('norm2')
plt.xlabel('iterations')
plt.title('trend of norm2 value')
plt.legend(['norm2 calculated by line search ','using analytical solution norm'])

print('----------------------------------------')
#方法3：后退线性搜索
#设置ba是α，bb是β和t
ba=0.01
bb=0.1
t=1
#设定初始迭代点
xf2=np.matrix([[1],[1],[1]])
x2=xf2
cnt=0
result2=[]
start=time.time_ns()
while True:
    while 1/2*np.linalg.norm(A*(x2-t*AT*(A*x2-b.T))-b.T,ord=2)>1/2*np.linalg.norm(A*x2-b.T,ord=2)-ba*t*(AT*(A*x2-b.T)).T*(AT*(A*x2-b.T)):
        t=bb*t
    cnt+=1
    x2=x2-t*AT*(A*x2-b.T)
    result2.append(1/2*np.linalg.norm(A*x2-b.T,ord=2))
    if 1/2*np.linalg.norm(A*x2-b.T,ord=2)-1/2*np.linalg.norm(A*(x2-t*AT*(A*x2-b.T))-b.T)<0.00001:
        break
print('使用后退搜索法的时间为：\n',time.time_ns()-start)
print('使用后退搜索法的迭代次数为：\n',cnt)
print('使用后退搜索法求出的系数（解）为：\n',x2)
print('使用后退搜索法求出的范数最小值为：\n',1/2*np.linalg.norm(A*x2-b.T,ord=2))
#同样设定横坐标
label2=[i for i in range(cnt)]
ax2=p.add_subplot(2,2,2)
plt.scatter(label2,result2,color='y')
plt.axhline(1/2*np.linalg.norm(A*xa-b.T,ord=2),color='r')
plt.ylabel('norm2')
plt.xlabel('iterations')
plt.title('trend of norm2 value')
plt.legend(['norm2 calculated by backtracking line search ','using analytical solution norm'])

ax3=p.add_subplot(2,2,3)
plt.scatter(label,result,color='b')
plt.scatter(label2,result2,color='y')
plt.axhline(1/2*np.linalg.norm(A*xa-b.T,ord=2),color='r')
plt.ylabel('norm2')
plt.xlabel('iterations')
plt.title('trend of norm2 value')
plt.legend(['norm2 calculated by line search ','using backtracking line search','using analytical solution norm'])
plt.show()