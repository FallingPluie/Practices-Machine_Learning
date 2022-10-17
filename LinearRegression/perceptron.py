import numpy as np
import matplotlib.pyplot as plt
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import dataCreator as dc
import model

lst,tst = dc.cLinear(5)
A1 = model.perceptron()
x = np.array([i[0] for i in lst])
y = np.array([i[1] for i in lst])
#x = np.array([(1,2),(1,4),(3,5),(6,5),(3,2),(5,4)])
#y = np.array([1,1,1,-1,-1,-1])
A1.train(x,y)
w,b = A1.get()
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
plt.plot(xp,yp)
plt.show()