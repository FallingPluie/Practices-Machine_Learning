import numpy as np
import packages.dataGenerator as dg
import packages.model as model

d1 = dg.linear()
lst,tst = d1.classify(20,5,label = True)
x = np.array([i[0] for i in lst])
y = np.array([[i[1]] for i in lst])
a1 = model.backpropagation()
a1.defualt(2,3,1)
a1.yp(x[0])
a1.train(x,y,10000)

for i in range(len(y)):
    _,yy = a1.yp(x[i])
    print("{0} {1}".format(y[i],yy))
print("End")

x = np.array([i[0] for i in tst]) #type: ignore
y = np.array([[i[1]] for i in tst]) #type: ignore
for i in range(len(y)):
    _,yy = a1.yp(x[i])
    print("{0} {1}".format(y[i],yy))
