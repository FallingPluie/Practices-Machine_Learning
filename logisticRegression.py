import packages.model as model
import packages.dataGenerator as dg
import numpy as np
import matplotlib.pyplot as plt

d1 = dg.linear()
lst,tst = d1.logistic(15)
a1 = model.logistic(1)
x = np.array([i[0] for i in lst])
y = np.array([i[1] for i in lst])
a1.train(x,y,50000)

w,b = a1.get()
for i in range(len(x)):
    if y[i] == 0:
        mark = "x"
        Color = "red"
    else:
        mark = "+"
        Color = "green"
    plt.scatter(x[i][0],x[i][1],marker = mark, color = Color)  # type: ignore
xp = np.linspace(0,10)
yp = - w[0]/w[1] * xp - b/w[1]
yd = xp * d1.w + d1.b
plt.plot(xp,yp,color = "blue")
plt.plot(xp,yd,color = "black") # 仅参考，并非最优分界线
plt.show()