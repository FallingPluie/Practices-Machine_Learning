import numpy as np
import random
import time
from math import sqrt

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

class backpropagation:
    def __init__(self,n1:float = 0.5,n2:float = 0.5) -> None:
        """
        创建未初始化单隐藏层BP神经网络

        Augs:
            n1 (float): 隐藏层到输出层学习率, (0,1)
            n2 (float): 输入层到隐藏层学习率, (0,1)
        """
        self.__initialized = 0
        self.n1 = n1
        self.n2 = n2

    def defualt(self,d:int,q:int,l:int) -> None:
        """
        随机初始化所有权重与阈值

        Args:
            d (int): 每个训练集中输入层单元的数量
            q (int): 每个训练集中隐藏层单元的数量
            l (int): 每个训练集中输出层单元的数量

            通常取q = sqrt(d + l) + c, 其中c为[1,10]任意实数
        """
        self.__v = np.random.random((q,d))
        self.__t = np.random.random(q)
        self.__w = np.random.random((l,q))
        self.__b = np.random.random(l)
        self.__d = d
        self.__q = q
        self.__l = l
        self.__initialized = 1

    def initialize(self,v:np.ndarray,t:np.ndarray,w:np.ndarray,b:np.ndarray) -> None:
        """
        自定义初始化模型权重与阈值

        Augs:
            v (np.ndarray): 输入层到隐藏层的权重, 二维数组 array[q][d]
            t (np.ndarray): 隐藏层的阈值, 一维数组 array[q]
            w (np.ndarray): 隐藏层到输出层的权重, 二维数组 array[l][q]
            b (np.ndarray): 输出层的阈值, 一维数组 array[l]

        其中:
            d: 每个训练集中输入层单元的数量
            q: 每个训练集中隐藏层单元的数量
            l: 每个训练集中输出层单元的数量

            通常取q = sqrt(d + l) + c, 其中c为[1,10]任意实数
        """
        try:
            tq,_ = t.shape
            bl,_ = b.shape
            print("(model.backpropagation.initialize)初始化失败, 阈值非一维数组, 检查输入数据")
        except:
            try:
                vq,vd = v.shape
                tq = t.shape[0]
                wl,wq = w.shape
                bl = b.shape[0]
                if vq == tq == wq and wl == bl:
                    self.__v = v
                    self.__t = t
                    self.__w = w
                    self.__b = b
                    self.__d = vd
                    self.__q = vq
                    self.__l = wl
                    self.__initialized = 1
                else:
                    print("(model.backpropagation.initialize)初始化失败, 数据规格不互相匹配, 检查输入数据")
            except:
                print("(model.backpropagation.initialize)初始化失败, 数据规格不符, 检查输入数据")

    def fx(self,z:float) -> np.ndarray:
        """
        logistic函数 g(z)
        """
        if z > 0:
            return 1/(1+np.exp(-z))
        else:
            return np.exp(z)/(np.exp(z)+1) # 避免因exp(z)过大导致数据溢出

    def hyout(self,x:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
        计算求得模型隐藏层与输出层的输出值

        Args:
            x (np.ndarray): 单个训练例的全部输入x, 一维矩阵

        Returns:
            tuple[np.ndarray,np.ndarray]: hout, yout
                hout: 该训练例的隐藏层全部输出
                yout: 该训练例的输出层全部输出
        """
        if self.__initialized == 0:
            print("未初始化模型")
            return np.zeros(1),np.zeros(1)
        alpha = np.array([np.dot(self.__v[i],x) for i in range(self.__q)])
        h = np.array([self.fx(alpha[i] - self.__t[i]) for i in range(self.__q)])
        beta = np.array([np.dot(h,self.__w[i]) for i in range(self.__l)])
        yout = np.array([self.fx(beta[i] - self.__b[i]) for i in range(self.__l)])# y预测输出
        return h,yout

    def __update(self,x:np.ndarray,y:np.ndarray) -> None:
        """
        迭代公式, 梯度下降法

        Args:
            x (np.ndarray): 单个训练例的全部输入
            y (np.ndarray): 单个训练例的全部输出
        """
        h,yout = self.hyout(x)
        g = np.array([yout[i]*(1-yout[i])*(y[i]-yout[i]) for i in range(self.__l)])
        e = np.array([(h[i]*(1-h[i]) * np.sum(np.array([g[j]*self.__w[j][i] for j in range(self.__l)]))) for i in range(self.__q)])
        # g,e公式见周志华《机器学习》.清华大学出版社.2016 p103
        for i in range(self.__l):
            for j in range(self.__q):
                self.__w[i][j] = self.__w[i][j] + self.n1*g[i]*h[j]
                for s in range(self.__d):
                    self.__v[j][s] = self.__v[j][s] + self.n2*e[j]*x[s]
                self.__t[j] = self.__t[j] - self.n2*e[j]
            self.__b[i] = self.__b[i] - self.n1*g[i]

    def __fit(self,x:np.ndarray,y:np.ndarray,n:int) -> None:
        """
        训练BP神经网络

        Args:
            x (np.ndarray): 输入集, 二维矩阵
            y (np.ndarray): 输出集, 二维矩阵
            n (int): 训练次数
        """
        progress = "="
        tper = n/20
        print("[{0:<20}], eta:          ".format(""),end = "")
        t1 = time.perf_counter()
        ### 以上部分为进度条
        index = list(range(len(x)))
        for count in range(n):
            random.shuffle(index)
            for i in index: # type: ignore
                self.__update(x[i],y[i])
        ### 以下部分为进度条
            s = (count+1)/tper-len(progress)
            t2 = time.perf_counter()
            t = str(round((n-count-1)*(t2 - t1)/(count+1),1))+'s'
            if s > 1e-3:
                print("\r[{0:<20}], eta:{1:<10}".format(progress,t),end = "")
                for i in range(int(s)+1):progress += "="
            else:
                print("\b\b\b\b\b\b\b\b\b\b\b\b\b\beta:{0:<10}".format(t),end="")
        print("\r[{0:=<20}], 已完成           ".format("="))

    def train(self,x:np.ndarray,y:np.ndarray,n:int = 1000, s_fold:int = 1) -> None:
        """
        训练BP神经网络

        Args:
            x (np.ndarray): 输入集, 二维矩阵
            y (np.ndarray): 输出集, 二维矩阵
            n (int): 每次验证训练次数, 默认为1000
            s_fold (int): s折交叉验证分割子集数, 小于等于1则不分割
        """
        # 主要为检查数据合法性与完成s折交叉验证
        try:
            xm,d = x.shape
            ym,l = y.shape
        except:
            print("(model.backpropagation.train)数据规格不符")
            return None
        if xm != ym:
            print("(model.backpropagation.train)训练例特征与标签数不等, 终止运行")
            return None
        if self.__initialized == 0:
            q = int(sqrt(d*l))+2
            print("(model.backpropagation.train)未初始化, 以 隐藏层单元数量: {0} 随机初始化所有权重与阈值".format(q))
            self.defualt(d,q,l)
        elif d != self.__d or l != self.__l:
            print("(model.backpropagation.train)输入x或y大小与初始化值不符, 现以 输入层单元数量: {0}, 隐藏层单元数量: {1}, 输出层单元数量: {2} 随机初始化所有权重与阈值".format(d,self.__q,l))
            self.defualt(d,self.__q,l)

        if s_fold <= 1:
            self.__fit(x,y,n)
        else:
            loss = -1
            step = int(len(x)/s_fold)
            state = np.random.get_state()
            np.random.shuffle(x)
            np.random.set_state(state)
            np.random.shuffle(y)
            temp_v,temp_t,temp_w,temp_b = self.__v, self.__t, self.__w, self.__b
            for i in range(s_fold):
                print("第 {0}/{1} 次:".format(i+1,s_fold))
                tstx,tsty = x[(s_fold-i-1)*step:(s_fold-i)*step],y[(s_fold-i-1)*step:(s_fold-i)*step]
                tx = np.delete(x,np.s_[(s_fold-i-1)*step:(s_fold-i)*step],0)
                ty = np.delete(y,np.s_[(s_fold-i-1)*step:(s_fold-i)*step],0)
                self.__fit(tx,ty,n)
                evaluate = self.evaluate(tstx,tsty)
                if loss < 0 or loss > evaluate:
                    temp_v,temp_t,temp_w,temp_b = self.__v, self.__t, self.__w, self.__b
                    loss = evaluate
            self.__v, self.__t, self.__w, self.__b = temp_v,temp_t,temp_w,temp_b

    def get(self) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        输出模型的全部权重与阈值

        Returns:
            tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: v, t, w, b
                v: 输入层到隐藏层的权重, 二维数组
                t: 隐藏层的阈值, 一维数组
                w: 隐藏层到输出层的权重, 二维数组
                b: 输出层的阈值, 一维数组
        """
        if self.__initialized == 1: return self.__v,self.__t,self.__w,self.__b
        else:
            print("(model.backpropagation.get)未初始化模型")
            return np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    def evaluate(self,x:np.ndarray,y:np.ndarray) -> float:
        """
        评估当前模型对某一测试集的平方差和

        Args:
            x (np.ndarray): 输入集, 二维矩阵
            y (np.ndarray): 输出集, 二维矩阵

        Returns:
            float: 损失, 平方差和
        """
        try:
            xm,d = x.shape
            ym,l = y.shape
        except:
            print("(model.backpropagation.evaluate)数据规格不符")
            return -1
        if xm != ym:
            print("(model.backpropagation.evaluate)训练例特征与标签数不等")
            return -1
        elif d != self.__d:
            print("(model.backpropagation.evaluate)输入单元数与模型不匹配")
            return -1
        elif l != self.__l:
            print("(model.backpropagation.evaluate)输出单元数与模型不匹配")
            return -1

        loss = 0
        for i in range(len(x)):
            _,y0 = self.hyout(x[i])
            for j in range(len(y0)):
                loss += pow(y[i][j]-y0[j],2)
        return loss