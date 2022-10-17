import random
from typing import Optional

def linear(n:int,delta:float = 1,test:bool = True)->tuple[list,Optional[list]]:
    """
    产生一个用于单变量线性回归的数据集

    Args:
        n (int): 点数量
        delta(float): 数据的上下变动范围
        test(bool): 是否产生一个同等大小的测试集

    Returns:
        tuple[list,Optional[list]]: (lst,tst)
            lst: 数据集，[(x_1,y_1),(x_2,y_2),……,(x_n,y_n)]
            tst: 测试集，[(x_1,y_1),(x_2,y_2),……,(x_n,y_n)]
    """
    w1 = random.randrange(-100,100)/10
    w0 = random.randrange(-100,100)/10
    lst = []
    for i in range(n):
        x1 = random.randrange(0,10000)/100
        x2 = random.randrange(int(x1*w1+w0*100-delta*100),int(x1*w1+w0*100+delta*100))/100
        lst.append((x1,x2))
    if test == False:
        return lst,None
    else:
        tst = []
        i = 0
        while i < n:
            x1 = random.randrange(0,10000)/100
            x2 = random.randrange(int(x1*w1+w0*100-delta*100),int(x1*w1+w0*100+delta*100))/100
            if (x1,x2) not in lst:
                tst.append((x1,x2))
                i = i + 1
            else:
                continue
        return lst,tst

def cLinear(n:int,test:bool = True)->tuple[list,Optional[list]]:
    """
    产生一个用于线性分类的数据集

    Args:
        n (int): y = 0与y = 1的数据组各n个
        test(bool): 是否产生一个同等大小的测试集

    Returns:
        tuple[list,Optional[list]]: (lst,tst)
            lst: 数据集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_n,x2_n),y_n)]
            tst: 测试集，[((x1_1,x2_1),y_1),((x1_2,x2_2),y_2),……,((x1_n,x2_n),y_n)]
    """
    w1 = random.randrange(-100,100)/10
    w0 = random.randrange(-100,100)/10
    lst = []
    for i in range(n):
        x1 = random.randrange(0,1000)/100
        x2 = random.randrange(int(x1*w1+w0)*100+100,15000)/100
        lst.append(((x1,x2),1))
    for i in range(n):
        x1 = random.randrange(0,1000)/100
        x2 = random.randrange(-15000,int(x1*w1+w0)*100)/100
        lst.append(((x1,x2),0))
    if test == False:
        return lst,None
    else:
        tst = []
        i = 0
        while i < n:
            x1 = random.randrange(0,1000)/100
            x2 = random.randrange(int(x1*w1+w0)*100+100,15000)/100
            if ((x1,x2),1) not in lst:
                tst.append(((x1,x2),1))
                i = i + 1
            else:
                continue
        i = 0
        while i < n:
            x1 = random.randrange(0,1000)/100
            x2 = random.randrange(-15000,int(x1*w1+w0)*100)/100
            if ((x1,x2),0) not in lst:
                tst.append(((x1,x2),0))
                i = i + 1
            else:
                continue
        return lst,tst

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    lst,tst = cLinear(30)
    for i in range(len(lst)):
        if lst[i][1] == 0:
            mark = "x"
            Color = "red"
        else:
            mark = "+"
            Color = "green"
        plt.scatter(lst[i][0][0],lst[i][0][1],marker = mark, color = Color)  # type: ignore
    plt.show()
    os.system("pause")
    try:
        for i in range(len(tst)):  # type: ignore
            if tst[i][1] == 0:  # type: ignore
                mark = "x"
                Color = "red"
            else:
                mark = "+"
                Color = "green"
            plt.scatter(tst[i][0][0],tst[i][0][1],marker = mark, color = Color)  # type: ignore
        plt.show()
    except:
        pass