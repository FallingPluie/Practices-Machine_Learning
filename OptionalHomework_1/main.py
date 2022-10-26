import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#logistic回归模型，与model中相同 (Update#4)
class logistic:
    def __init__(self,learn_rate:float = 1.0) -> None:
        """
        创建logistic回归模型, 学习率(learn_rate)默认为1, 应大于0
        """
        self.__w = np.array([0,0,0]) # w1,w2,b
        self.learn_rate = learn_rate

    def fx(self,w:np.ndarray,x:np.ndarray) -> np.ndarray:
        """
        logistic函数 g(w·x)
        """
        z = np.dot(w,x)
        if z > 0:
            return 1/(1+np.exp(-z))
        else:
            return np.exp(z)/(np.exp(z)+1) # 避免因exp(z)过大导致数据溢出

    def __update(self,x:np.ndarray,y:float) -> None:
        """
        迭代公式, 选取损失函数为L(w)=sum((y-logistic(w·x))^2), 梯度下降法

        Args:
            x (np.ndarray): 单组坐标
            y (float): 该组标签
        """
        a = np.array([x[0],x[1],1])
        self.__w = self.__w+self.learn_rate*(y-self.fx(self.__w,a))*self.fx(self.__w,a)*(1-self.fx(self.__w,a))*a

    def train(self,x:np.ndarray,y:np.ndarray,n:int = 50000) -> None:
        """
        训练模型

        Args:
            x (np.ndarray): 特征集, 形如 [(x1_1,x2_1),(x1_2,x2_2),……,(x1_n,x2_n)]
            y (np.ndarray): 标签集, 形如 [y_1,y_2,y_3,……,y_n]
            n(int): 训练次数, 默认为50000
        """
        if len(x) != len(y):
            print("feature do not match with label in quantity")
            return None
        index = list(range(len(x)))
        for count in range(n):
            random.shuffle(index)
            for i in index:
                self.__update(x[i],y[i])
            if (count+1)%2500 == 0: print(count+1) # 报数

    def get(self) -> tuple[list, float]:
        """
        Returns:
            tuple[list,float]: (w,b)
                w:权重
                b:偏置
        """
        return [self.__w[0],self.__w[1]],self.__w[2]

ip = "OptionalHomework_1\\prob2.2 train Input.asc"
dp = "OptionalHomework_1\\prob2.2 train Desired.asc"
tip = "OptionalHomework_1\\prob2.2 test Input.asc"
tdp = "OptionalHomework_1\\prob2.2 test Desired.asc"

x = pd.read_csv(ip,sep = "\t").to_numpy()
y = pd.read_csv(dp,sep = "\t").to_numpy()
tx = pd.read_csv(tip,sep = "\t").to_numpy()
ty = pd.read_csv(tdp,sep = "\t").to_numpy()

#选用logistic回归
core = logistic()
core.train(x,y,10000)
w,b = core.get()

#作图
plt.subplot(1,2,1)
plt.title("train")
for i in range(len(x)):
    if y[i] == 0:
        mark = "x"
        Color = "red"
    else:
        mark = "+"
        Color = "green"
    plt.scatter(x[i][0],x[i][1],marker = mark, color = Color)  # type: ignore
xp = np.linspace(-1,1)
yp = - w[0]/w[1] * xp - b/w[1]
plt.plot(xp,yp)

plt.subplot(1,2,2)
plt.title("text")
for i in range(len(tx)):
    if ty[i] == 0:
        mark = "x"
        Color = "red"
    else:
        mark = "+"
        Color = "green"
    plt.scatter(tx[i][0],tx[i][1],marker = mark, color = Color)  # type: ignore
plt.plot(xp,yp)

plt.show()