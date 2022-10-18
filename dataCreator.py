import random
from typing import Optional

class linear():
    def __init__(self,xmin:int = 0, xmax:int = 10) -> None:
        """
        生成一条用于产生各类线性数据的直线 y = w * x + b; w,b 默认在-10到10之间选取

        Args:
            xmax(int): x最大取值范围
            xmin(int): x最小取值范围
        """
        self.w:float = random.randrange(-100,100)/10
        self.b:float = random.randrange(-100,100)/10
        self.xmax:int = xmax
        self.xmin:int = xmin

    def regression(self,n:int,delta:float = 1,test:int = 0)->tuple[list,Optional[list]]:
        """
        产生一个用于单变量线性回归的数据集

        Args:
            n (int): 点数量
            delta(float): 数据的上下变动范围
            test(int): 测试集点数量, 不大于0则返回None

        Returns:
            tuple[list,Optional[list]]: (lst,tst)
                lst: 数据集，[(x_1,y_1),(x_2,y_2),……,(x_n,y_n)]
                tst: 测试集，[(x_1,y_1),(x_2,y_2),……,(x_m,y_m)]
        """
        if self.xmax <= self.xmin:
            print ("illigal x rage")
            return [0],None
        lst = []
        for i in range(n):
            x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x2 = random.randrange(int(x1*self.w+self.b-delta)*100,int(x1*self.w+self.b+delta)*100)/100
            lst.append((x1,x2))
        if test <= 0:
            return lst,None
        else:
            tst = []
            i = 0
            while i < test:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(x1*self.w+self.b-delta)*100,int(x1*self.w+self.b+delta)*100)/100
                if (x1,x2) not in lst:
                    tst.append((x1,x2))
                    i = i + 1
                else:
                    continue
            return lst,tst

    def classify(self,n:int,test:int = 0,label:bool = False)->tuple[list,Optional[list]]:
        """
        产生一个线性可分的数据集

        Args:
            n (int): 两种标签的数据组各n个
            test(int): 测试集,两种标签的数据组各test个, test <= 0则不产生测试集
            label(bool): False: 标签为{-1, 1}.  True: 标签为{0, 1}

        Returns:
            tuple[list,Optional[list]]: (lst,tst)
                lst: 数据集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_n,x2_n),y_n)]
                tst: 测试集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_m,x2_m),y_m)]
        """
        if self.xmax <= self.xmin:
            print ("illigal x rage")
            return [0],None
        if label == False: tag = -1
        else: tag = 0
        if self.w >= 0:
            posmax = self.xmax
            posmin = self.xmin
        else:
            posmax = self.xmin
            posmin = self.xmax
        lst = []
        for i in range(n):
            x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x2 = random.randrange(int(x1*self.w+self.b+5)*100,int(posmax*self.w+self.b+50)*100)/100
            lst.append(((x1,x2),1))
            x_1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x_2 = random.randrange(int(posmin*self.w+self.b-50)*100,int(x_1*self.w+self.b-5)*100)/100
            lst.append(((x_1,x_2),tag))
        if test <= 0:
            return lst,None
        else:
            tst = []
            i = 0
            while i < test:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(x1*self.w+self.b+5)*100,int(posmax*self.w+self.b+50)*100)/100
                if ((x1,x2),1) not in lst:
                    tst.append(((x1,x2),1))
                    i = i + 1
                else:
                    continue
            i = 0
            while i < test:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(posmin*self.w+self.b-50)*100,int(x1*self.w+self.b-5)*100)/100
                if ((x1,x2),tag) not in lst:
                    tst.append(((x1,x2),tag))
                    i = i + 1
                else:
                    continue
            return lst,tst

    def logistic(self,n:int,delta:int = 15,test:int = 0,label:bool = True)->tuple[list,Optional[list]]:
        """
        产生一个不完全线性可分的数据集

        Args:
            n (int): 两种标签的数据组各n个
            delta(int): 允许异种标签超出原始绘图线的最大距离
            test(int): 测试集两种标签的数据组各test个, test <= 0则不产生测试集
            label(bool): False: 标签为{-1, 1}.  True: 标签为{0, 1}

        Returns:
            tuple[list,Optional[list]]: (lst,tst)
                lst: 数据集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_n,x2_n),y_n)]
                tst: 测试集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_m,x2_m),y_m)]
        """
        if self.xmax <= self.xmin:
            print ("illigal x rage")
            return [0],None
        n1 = int (0.2 * n)
        n0 = n - n1
        if label == False: tag = -1
        else: tag = 0
        if self.w >= 0:
            posmax = self.xmax
            posmin = self.xmin
        else:
            posmax = self.xmin
            posmin = self.xmax
        lst = []
        for i in range(n0):
            x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x2 = random.randrange(int(x1*self.w+self.b)*100,int(posmax*self.w+self.b+50)*100)/100
            lst.append(((x1,x2),1))
            x_1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x_2 = random.randrange(int(posmin*self.w+self.b-50)*100,int(x_1*self.w+self.b)*100)/100
            lst.append(((x_1,x_2),tag))
        for i in range(n1):
            x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x2 = random.randrange(int(x1*self.w+self.b)*100,int(x1*self.w+self.b+delta)*100)/100
            lst.append(((x1,x2),tag))
            x_1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
            x_2 = random.randrange(int(x_1*self.w+self.b-delta)*100,int(x_1*self.w+self.b)*100)/100
            lst.append(((x_1,x_2),1))
        if test <= 0:
            return lst,None
        else:
            tst = []
            t1 = int(0.2 * test)
            t0 = test - t1
            i = 0
            while i < t0:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(x1*self.w+self.b)*100,int(posmax*self.w+self.b+50)*100)/100
                if ((x1,x2),1) not in lst:
                    tst.append(((x1,x2),1))
                    i = i + 1
                else:
                    continue
            i = 0
            while i < t0:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(posmin*self.w+self.b-50)*100,int(x1*self.w+self.b)*100)/100
                if ((x1,x2),tag) not in lst:
                    tst.append(((x1,x2),tag))
                    i = i + 1
                else:
                    continue
            i = 0
            while i < t1:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(x1*self.w+self.b)*100,int(x1*self.w+self.b+delta)*100)/100
                if ((x1,x2),tag) not in lst:
                    tst.append(((x1,x2),tag))
                    i = i + 1
                else:
                    continue
            i = 0
            while i < t1:
                x1 = random.randrange(self.xmin * 100,self.xmax * 100)/100
                x2 = random.randrange(int(x1*self.w+self.b-delta)*100,int(x1*self.w+self.b)*100)/100
                if ((x1,x2),1) not in lst:
                    tst.append(((x1,x2),1))
                    i = i + 1
                else:
                    continue
            return lst,tst

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    dg = linear(10,20)
    lst,tst = dg.logistic(10,test = 10,label = False)
    for i in range(len(lst)):
        if lst[i][1] == -1:
            mark = "x"
            Color = "red"
        else:
            mark = "+"
            Color = "green"
        plt.scatter(lst[i][0][0],lst[i][0][1],marker = mark, color = Color)  # type: ignore
    x = np.linspace(dg.xmin,dg.xmax)
    y = x * dg.w + dg.b
    plt.plot(x,y)
    plt.show()
    os.system("pause")
    try:
        for i in range(len(tst)):  # type: ignore
            if tst[i][1] == -1:  # type: ignore
                mark = "x"
                Color = "red"
            else:
                mark = "+"
                Color = "green"
            plt.scatter(tst[i][0][0],tst[i][0][1],marker = mark, color = Color)  # type: ignore
        plt.plot(x,y)
        plt.show()
    except:
        pass