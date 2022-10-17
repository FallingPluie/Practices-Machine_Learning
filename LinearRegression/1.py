import matplotlib.pyplot as plt
import numpy as np
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import dataCreator as dc

# 初始化
lst,_ = dc.linear(20,test=False)
n = len(lst)

# 最小二乘法
sumx = 0
for i in range(n):
    sumx = sumx + lst[i][0]
avrg_x = sumx/n
sum1 = sum2 = 0
for i in range(n):
    sum1 = sum1 + ((lst[i][0] - avrg_x) * lst[i][1])
    sum2 = sum2 + lst[i][0] * lst[i][0]

w = sum1/(sum2 - sumx * sumx / n)

sum3 = 0
for i in range(n):
    sum3 = sum3 + (lst[i][1] - w * lst[i][0])

b = sum3/n

# 画图
fig = plt.figure()
ax1 = fig.add_subplot(1,1, 1)
for i in range(n):
    ax1.scatter(lst[i][0],lst[i][1],marker = "x", color = "green")  # type: ignore
x = np.linspace(0,100)
y = x * w + b
ax1.plot(x,y)
plt.show()