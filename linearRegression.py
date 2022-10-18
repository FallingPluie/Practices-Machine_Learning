import matplotlib.pyplot as plt
import numpy as np
import packages.dataGenerator as dg

# 初始化
d1 = dg.linear()
lst,_ = d1.regression(20)
n = len(lst)

# 最小二乘法
sumx = np.sum([i[0] for i in lst])
avrg_x = sumx/n
sum1 = np.sum([(i[0]-avrg_x)*i[1] for i in lst])
sum2 = np.sum([i[0]*i[0] for i in lst])
w = sum1/(sum2 - sumx * sumx / n)
sum3 = np.sum([(i[1]-w*i[0]) for i in lst])
b = sum3/n

# 画图
for i in range(n):
    plt.scatter(lst[i][0],lst[i][1],marker = "x", color = "red")  # type: ignore
x = np.linspace(d1.xmin,d1.xmax)
y = x * w + b
plt.plot(x,y)
plt.show()