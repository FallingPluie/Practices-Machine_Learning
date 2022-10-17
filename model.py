import numpy as np

class perceptron:
    def __init__(self, learn_rate:float = 1.0) -> None:
        """
        创建感知机, 学习率(learn_rate)默认为1.0, 应大于0
        """
        self.__w = np.array([0,0])
        self.__b = 0.0
        self.learn_rate = learn_rate

    def __update(self,x:np.ndarray,y:int) -> None:
        """
        迭代公式

        Args:
            x (np.ndarray): 单组特征
            y (int): 该组标签
        """
        self.__w = self.__w + self.learn_rate * x * y
        self.__b = self.__b + self.learn_rate * y

    def judge(self,x:np.ndarray,y:int) -> int:
        """
        误分类判据

        Args:
            x (np.ndarray): 单组特征
            y (int): 该组标签

        Returns:
            int: 感知机是否误分类(是: -1, 否: 1, 点位于超平面上: 0)
        """
        return np.sign(y * (np.dot(self.__w, x) + self.__b))

    def train(self,x:np.ndarray,y:np.ndarray,n:int = -1) -> None:
        """
        训练感知机

        Args:
            x (np.ndarray): 特征集, 形如 [(x1_1,x2_1),(x1_2,x2_2),……,(x1_n,x2_n)]
            y (np.ndarray): 标签集, 形如 [y_1,y_2,y_3,……,y_n]
            n(int): 训练次数上限, 小于等于0则训练直至收敛
        """
        if len(x) != len(y):
            print("feature do not match with label in quantity")
            return None
        run = 1
        if n <= 0:
            while run:
                counts = len(x)
                for i in range(len(x)):
                    if self.judge(x[i],y[i]) <= 0:
                        self.__update(x[i],y[i])
                    else:
                        counts = counts - 1
                if counts == 0: run = 0
        else:
            while run and n > 0:
                counts = len(x)
                for i in range(len(x)):
                    if self.judge(x[i],y[i]) <= 0:
                        self.__update(x[i],y[i])
                    else:
                        counts = counts - 1
                if counts == 0: run = 0
                n = n - 1
            if n == 0 and run != 0: print("达到训练次数上限, 已停止训练, 未收敛")

    def get(self) -> tuple[np.ndarray,float]:
        """
        Returns:
            tuple[np.ndarray,float]: (w,b)
                w:权重
                b:偏置
        """
        return self.__w,self.__b