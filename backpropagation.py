import numpy as np
import packages.dataGenerator as dg
import packages.model as model

d1 = dg.linear(-1,1)
lst,tst = d1.classify(20,5,label = True)
x = np.array([[i[0][0],i[0][1]/10] for i in lst])
y = np.array([[i[1]] for i in lst])
a1 = model.backpropagation(decay_rate=1)
a1.defualt(2,3,1)
a1.train(x,y,1500,3)

count1 = 0
for i in range(len(y)):
    _,yy = a1.hyout(x[i])
    print("{0} {1}".format(y[i],yy[0]))
    if y[i]-yy < 1e-2:
        count1 += 1
print(count1/len(y))

x = np.array([i[0] for i in tst]) #type: ignore
y = np.array([[i[1]] for i in tst]) #type: ignore
print("对测试集平方差和:{0:.5f}".format(a1.evaluate(x,y)))

'''for i in range(len(y)):
    _,yy = a1.hyout(x[i])
    print("{0} {1}".format(y[i],yy))'''
