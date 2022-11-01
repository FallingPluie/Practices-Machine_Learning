import numpy as np
import packages.model as model
import random

def data(n:int) -> tuple[np.ndarray,np.ndarray]:
    lst = []
    for i in range(n):
        x11 = random.randrange(500,1000)/1000
        x12 = random.randrange(500,1000)/1000
        lst.append(((x11,x12),1))
        x21 = random.randrange(500,1000)/1000
        x22 = random.randrange(-500,-1000,-1)/1000
        lst.append(((x21,x22),2))
        x31 = random.randrange(-500,-1000,-1)/1000
        x32 = random.randrange(-500,-1000,-1)/1000
        lst.append(((x31,x32),3))
        x41 = random.randrange(-500,-1000,-1)/1000
        x42 = random.randrange(500,1000)/1000
        lst.append(((x41,x42),4))
    random.shuffle(lst)
    x = np.array([i[0] for i in lst])
    y = np.array([i[1] for i in lst])
    return x,y

num = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
x,y = data(10)
y1 = np.array([num[i-1] for i in y])
core = model.backpropagation()
core.defualt(2,10,4)

core.train(x,y1,3000,1)

print(core.evaluate(x,y1))

for i in range(10):
    print(x[i],y[i],np.argmax(core.hyout(x[i])[1])+1)