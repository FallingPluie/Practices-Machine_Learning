print("初始化……")
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import packages.model as model
import pickle

# 初始化数据
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255 # 压缩数据
le = len(x_train)
x_train_1 = np.reshape(x_train,[le,784])
le = len(x_test)
x_test_1 = np.reshape(x_test,[le,784])
#num = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])
num = np.array([[1,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]])
y_train = np.array([num[y_train[i]] for i in range(len(y_train))])
y_test = np.array([num[y_test[i]] for i in range(len(y_test))])

# 初始化模型
core = model.backpropagation(0.4,0.4,0.5)
core.defualt(784,35,10)
print("初始化完成")

'''start = 0
plt.figure(figsize=(10,10))
for i in range(start,start+25):
    plt.subplot(5,5,i+1-start)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary) # type: ignore
    plt.xlabel(np.argmax(core.hyout(x_train_1[i])[1])) # type: ignore
plt.show()'''

# 开始训练
'''with open('number.BP','rb') as file:
    core = pickle.load(file)'''
core.train(x_train_1,y_train,15)
'''with open('number.BP','wb') as file:
    pickle.dump(core,file)'''

start = 0
plt.figure(figsize=(10,10))
for i in range(start,start+25):
    plt.subplot(5,5,i+1-start)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary) # type: ignore
    plt.xlabel(np.argmax(core.hyout(x_test_1[i])[1])) # type: ignore
plt.show()