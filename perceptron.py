import numpy as np
import matplotlib.pyplot as plt
import packages.dataGenerator as dg
import packages.model as model

d1 = dg.linear()
lst,tst = d1.classify(5)
a1 = model.perceptron()
x = np.array([i[0] for i in lst])
y = np.array([i[1] for i in lst])
#x = np.array([(1,2),(1,4),(3,5),(6,5),(3,2),(5,4)])
#y = np.array([1,1,1,-1,-1,-1])
a1.train(x,y)
w,b = a1.get()
#print(w,b)
#print("{0}\n{1}".format(x,y))

for i in range(len(x)):
    if y[i] == -1:
        mark = "x"
        Color = "red"
    else:
        mark = "+"
        Color = "green"
    plt.scatter(x[i][0],x[i][1],marker = mark, color = Color)  # type: ignore
xp = np.linspace(0,10)
yp = - w[0]/w[1] * xp - b/w[1]
#yd = xp * d1.w + d1.b
plt.plot(xp,yp,color = "blue")
#plt.plot(xp,yd,color = "black")
plt.show()