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
N_GENERATIONS = 400                                 # 迭代次数
POP_SIZE = 100                                      # 种群大小
name = 'DTLZ1'                                      # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
M = 3                                               # 目标个数
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

# 绘制PF
if(M<=3):
    type2 = ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='r',marker = 'x',s=200)
    plt.legend((type1, type2), (u'Non-dominated solution', u'PF'))
else:
    fig1 = plt.figure()
    plt.xlim([0,M])
    for i in range(pop.shape[0]):
        plt.plot(np.array(pop[i,:]))    
plt.show()

#IGD
score = IGD(popfun,PF)