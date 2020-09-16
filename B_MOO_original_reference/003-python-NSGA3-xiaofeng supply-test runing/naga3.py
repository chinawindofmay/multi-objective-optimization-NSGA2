# -*- coding: utf-8 -*-
"""
程序功能：论文复现
论文信息：
An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach, Part I: Solving Problems with Box Constraint
作者：(晓风)wangchao
最初建立时间：2019.03.26
最近修改时间：2019.04.01
最小化问题：DTLZ1,DTLZ2,DTLZ3
NSGA3的简单实现
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from utils import uniformpoint,funfun,cal,GO,envselect,IGD
import copy
import random

#参数设置
N_GENERATIONS = 10                                 # 迭代次数
POP_SIZE = 10                                    # 种群大小
name = 'DTLZ1'                                      # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
M = 8                                               # 目标个数
t1 = 20                                             # 交叉参数t1
t2 = 20                                             # 变异参数t2
pc = 1                                              # 交叉概率
pm = 1                                              # 变异概率
#画图部分
if(M<=3):
    fig = plt.figure()
    ax = Axes3D(fig)

###################################################################################################################################################################
#产生一致性的参考点和随机初始化种群
Z,N = uniformpoint(POP_SIZE,M)#生成一致性的参考解
pop,popfun,PF,D = funfun(M,N,name)#生成初始种群及其适应度值，真实的PF,自变量个数
popfun = cal(pop,name,M,D)#计算适应度函数值
Zmin = np.array(np.min(popfun,0)).reshape(1,M)#求理想点
#ax.scatter(Z[:,0],Z[:,1],Z[:,2],c='r')
#ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='b')

#迭代过程
for i in range(N_GENERATIONS):
    print("第{name}次迭代".format(name=i))

    matingpool=random.sample(range(N),N)
    matingpool=np.random.randint(N, size=N)
    # test value
    matingpool=[5, 2, 6, 7, 3, 1, 4, 1]
    off = GO(pop[matingpool,:],t1,t2,pc,pm)#遗传算子,模拟二进制交叉和多项式变异
    offfun = cal(off,name,M,D)#计算适应度函数
    mixpop = copy.deepcopy(np.vstack((pop, off)))
    Zmin = np.array(np.min(np.vstack((Zmin,offfun)),0)).reshape(1,M)#更新理想点
    pop = envselect(mixpop,N,Z,Zmin,name,M,D)
    popfun = cal(pop,name,M,D)
    if(M<=3):
        ax.cla()
        type1 = ax.scatter(popfun[:,0],popfun[:,1],popfun[:,2],c='g')
        plt.pause(0.00001)

#test value
popfun=np.array([[0.001740539,0.000505777,1.220228085,0.08365739,0.979644224,26.99090338,32.33749364,154.5825837],
[0.330216274,2.028785642,0,8.643557572,0.943363137,2.223168159,82.66332125,37.56273832],
[0.488432283,0.187703277,0,0.054875526,0.087518438,19.20889645,198.0122316,0],
[0,0,26.31358908,2.852262931,19.4835994,9.05379306,78.6376131,0],
[0.306789973,1.857447174,7.533014447,0.915552994,36.73179559,8.810956753,78.2768302,0],
[1.586282531,7.488620221,0,0.984357766,1.35194207,2.12365533,121.0149938,0],
[1.138863793,7.973089515,0,0.982198583,1.356631393,2.131021403,121.4347442,0],
[0,0,1.863182801,0.238799603,1.313066638,79.25006072,112.6554837,0]])

# 绘制PF
if(M<=3):
    type2 = ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='r',marker = 'x',s=200)
    plt.legend((type1, type2), (u'Non-dominated solution', u'PF'))
else:
    fig1 = plt.figure()
    plt.xlim([0,M])

    for i in range(popfun.shape[0]):
        plt.plot(np.array(popfun[i,:]))
plt.show()

#IGD
score = IGD(popfun,PF)
print(score)