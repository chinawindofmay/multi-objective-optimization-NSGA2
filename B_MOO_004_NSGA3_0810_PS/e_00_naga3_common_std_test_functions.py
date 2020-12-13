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
import random
from scipy.special import comb
from itertools import combinations
import copy
import math

class NSGA3:
    def __init__(self,N_GENERATIONS,POP_SIZE,function_name,FUN_NUM,t1,t2,pc,pm):
        self.GENERATIONS = N_GENERATIONS  # 迭代次数
        self.POP_SIZE = POP_SIZE  # 种群大小
        self.FUNCTION_NAME = function_name  # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
        self.M = FUN_NUM  # 目标个数
        self.X_COUNT = FUN_NUM + 4  # 定义自变量个数为目标个数加4
        self.t1 = t1  # 交叉参数t1
        self.t2 = t2  # 变异参数t2
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率

    # 参考点
    def uniformpoint(self,N):
        H1 = 1
        while (comb(H1 + self.M - 1, self.M - 1) <= N):
            H1 = H1 + 1
        H1 = H1 - 1
        Z = np.array(list(combinations(range(H1 + self.M - 1), self.M - 1))) - np.tile(
            np.array(list(range(self.M - 1))), (int(comb(H1 + self.M - 1, self.M - 1)), 1))
        Z = (np.hstack((Z, H1 + np.zeros((Z.shape[0], 1)))) - np.hstack((np.zeros((Z.shape[0], 1)), Z))) / H1
        if H1 < self.M:
            H2 = 0
            while (comb(H1 + self.M - 1, self.M - 1) + comb(H2 + self.M - 1, self.M - 1) <= N):
                H2 = H2 + 1
            H2 = H2 - 1
            if H2 > 0:
                W2 = np.array(list(combinations(range(H2 + self.M - 1), self.M - 1))) - np.tile(
                    np.array(list(range(self.M - 1))), (int(comb(H2 + self.M - 1, self.M - 1)), 1))
                W2 = (np.hstack((W2, H2 + np.zeros((W2.shape[0], 1)))) - np.hstack(
                    (np.zeros((W2.shape[0], 1)), W2))) / H2
                W2 = W2 / 2 + 1 / (2 * self.M)
                Z = np.vstack((Z, W2))  # 按列合并
        Z[Z < 1e-6] = 1e-6
        N = Z.shape[0]
        return Z, N

    # 初始化
    def initial_pop_and_theory_pof(self, N):
        low = np.zeros((1, self.X_COUNT))
        up = np.ones((1, self.X_COUNT))
        # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
        pop = np.tile(low, (N, 1)) + (np.tile(up, (N, 1)) - np.tile(low, (N, 1))) * np.random.rand(N, self.X_COUNT)

        # 计算理论的POF，用于与结果做对照使用
        Z_theory, nouse = self.uniformpoint(N)
        if self.FUNCTION_NAME == 'DTLZ1':
            Z_theory = Z_theory / 2
        elif self.FUNCTION_NAME == 'DTLZ2':
            # P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
            Z_theory = Z_theory / np.tile(np.array(np.sqrt(np.sum(Z_theory ** 2, 1))).reshape(Z_theory.shape[0], 1), (1, self.M))
        elif self.FUNCTION_NAME == 'DTLZ3':
            # P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
            Z_theory = Z_theory / np.tile(np.array(np.sqrt(np.sum(Z_theory ** 2, 1))).reshape(Z_theory.shape[0], 1), (1, self.M))
        return pop,  Z_theory

    # fitness计算
    def fitness(self, pop):
        N = pop.shape[0]
        if self.FUNCTION_NAME == 'DTLZ1':
            # 参考书205页
            g = np.array(100 * (self.X_COUNT - self.M + 1 + np.sum(((pop[:, (self.M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (self.M - 1):] - 0.5))), axis=1))).reshape(N, 1)
            pop_fitness = np.multiply(
                0.5 * np.tile(1 + g, (1, self.M)),
                (np.fliplr(
                        (np.hstack(
                            (np.ones((g.shape[0], 1)),
                             pop[:, :(self.M - 1)])
                        )).cumprod(1)
                    )
                )
            )
            pop_fitness = np.multiply(
                pop_fitness,
                (np.hstack(
                    (np.ones((g.shape[0], 1)),
                     1 - np.fliplr(pop[:, :(self.M - 1)]))
                ))
            )
        elif self.FUNCTION_NAME == 'DTLZ2':
            g = np.array(np.sum((pop[:, (self.M - 1):] - 0.5) ** 2, 1)).reshape(N, 1)
            pop_fitness = np.multiply(np.tile(1 + g, (1, self.M)), (np.fliplr(
                (np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(self.M - 1)] * (np.pi / 2))))).cumprod(1))))
            pop_fitness = np.multiply(pop_fitness, (
                np.hstack((np.ones((g.shape[0], 1)),  np.sin(np.fliplr(pop[:, :(self.M - 1)]) * (np.pi / 2))))))

        elif self.FUNCTION_NAME == 'DTLZ3':
            g = np.array(100 * (self.X_COUNT - self.M + 1 + np.sum(
                ((pop[:, (self.M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (self.M - 1):] - 0.5))),
                1))).reshape(N, 1)
            pop_fitness = np.multiply(np.tile(1 + g, (1, self.M)), (np.fliplr(
                (np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(self.M - 1)] * (np.pi / 2))))).cumprod(1))))
            pop_fitness = np.multiply(pop_fitness, (
                np.hstack((np.ones((g.shape[0], 1)),  np.sin(np.fliplr(pop[:, :(self.M - 1)]) * (np.pi / 2))))))
        return pop_fitness

    # 模拟二进制交叉和多项式变异
    def crossover_and_mutation(self,pop, t1, t2, pc, pm):
        pop1 = copy.deepcopy(pop[0:int(pop.shape[0] / 2), :])
        pop2 = copy.deepcopy(pop[(int(pop.shape[0] / 2)):(int(pop.shape[0] / 2) * 2), :])
        N, D = pop1.shape[0], pop1.shape[1]
        # 模拟二进制交叉
        beta = np.zeros((N, D))
        mu = np.random.random_sample([N, D])
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (t1 + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (t1 + 1))
        beta = beta * ((-1) ** (np.random.randint(2, size=(N, D))))
        beta[np.random.random_sample([N, D]) < 0.5] = 1
        beta[np.tile(np.random.random_sample([N, 1]) > pc, (1, D))] = 1
        off = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))
        # 多项式变异
        low = np.zeros((2 * N, D))
        up = np.ones((2 * N, D))
        site = np.random.random_sample([2 * N, D]) < pm / D
        mu = np.random.random_sample([2 * N, D])
        temp = site & (mu <= 0.5)
        off[off < low] = low[off < low]
        off[off > up] = up[off > up]
        off[temp] = off[temp] + (up[temp] - low[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                    (1 - (off[temp] - low[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)) - 1)
        temp = site & (mu > 0.5)
        off[temp] = off[temp] + (up[temp] - low[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                    (1 - (up[temp] - off[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)))
        return off

    # 非支配排序
    def NDsort(self,mixpop, N):
        nsort = N  # 排序个数
        N, M = mixpop.shape[0], mixpop.shape[1]
        Loc1 = np.lexsort(mixpop[:, ::-1].T)  # loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
        mixpop2 = mixpop[Loc1]
        Loc2 = Loc1.argsort()  # loc2为旧矩阵元素在新矩阵中的位置
        frontno = np.ones(N) * (np.inf)  # 初始化所有等级为np.inf
        # frontno[0]=1#第一个元素一定是非支配的
        maxfno = 0  # 最高等级初始化为0
        while (np.sum(frontno < np.inf) < min(nsort, N)):  # 被赋予等级的个体数目不超过要排序的个体数目
            maxfno = maxfno + 1
            for i in range(N):
                if (frontno[i] == np.inf):
                    dominated = 0
                    for j in range(i):
                        if (frontno[j] == maxfno):
                            m = 0
                            flag = 0
                            while (m < M and mixpop2[i, m] >= mixpop2[j, m]):
                                if (mixpop2[i, m] == mixpop2[j, m]):  # 相同的个体不构成支配关系
                                    flag = flag + 1
                                m = m + 1
                            if (m >= M and flag < M):
                                dominated = 1
                                break
                    if dominated == 0:
                        frontno[i] = maxfno
        frontno = frontno[Loc2]
        return frontno, maxfno

    # 求两个向量矩阵的余弦值,x的列数等于y的列数
    def pdist(self,x, y):
        x0 = x.shape[0]
        y0 = y.shape[0]
        xmy = np.dot(x, y.T)  # x乘以y
        xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)
        ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
        xmmym = np.dot(xm, ym)
        cos = xmy / xmmym
        return cos

    # 临界面上进行选择
    def lastselection(self,popfun1, popfun2, K, Z, Zmin):
        # 选择最后一个front的解
        popfun = copy.deepcopy(np.vstack((popfun1, popfun2))) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))
        N, M = popfun.shape[0], popfun.shape[1]
        N1 = popfun1.shape[0]
        N2 = popfun2.shape[0]
        NZ = Z.shape[0]

        # 正则化
        extreme = np.zeros(M)
        w = np.zeros((M, M)) + 1e-6 + np.eye(M)
        for i in range(M):
            extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i, :], (N, 1))), 1))

        # 计算截距
        extreme = extreme.astype(int)  # python中数据类型转换一定要用astype
        try:  # 修改
            temp = np.linalg.inv(np.mat(popfun[extreme, :]))  # 逆矩阵
            # temp = np.linalg.pinv(np.mat(popfun[extreme,:]))   #广义逆矩阵 原来是求广义逆矩阵的，所以导致多个目标的时候可以求到广义逆矩阵。
        except:
            print("矩阵不存在逆矩阵")
            temp = np.full((M, M), np.nan)
        hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))
        a = 1 / hyprtplane
        if np.any(np.isnan(a)):  # 修改 原来的形式是：sum的形式 np.sum(a==math.nan) != 0
            a = np.max(popfun, 0)
        np.array(a).reshape(M, 1)  # 一维数组转二维数组
        a = a.T - Zmin  #修改
        # a = a.T
        popfun = popfun / (np.tile(a, (N, 1)))

        ##联系每一个解和对应向量
        # 计算每一个解最近的参考线的距离
        cos = self.pdist(popfun, Z)
        distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
        # 联系每一个解和对应的向量
        d = np.min(distance.T, 0)
        pi = np.argmin(distance.T, 0)

        # 计算z关联的个数
        rho = np.zeros(NZ)
        for i in range(NZ):
            rho[i] = np.sum(pi[:N1] == i)

        # 选出剩余的K个
        choose = np.zeros(N2)
        choose = choose.astype(bool)
        zchoose = np.ones(NZ)
        zchoose = zchoose.astype(bool)
        while np.sum(choose) < K:
            # 选择最不拥挤的参考点
            temp = np.ravel(np.array(np.where(zchoose == True)))
            jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
            j = temp[jmin[np.random.randint(jmin.shape[0])]]
            #        I = np.ravel(np.array(np.where(choose == False)))
            #        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
            # 修改，将下面的一行替换为两行
            kkk = np.where(pi[N1:] == j, 1, 0) * np.where(choose == 0, 1, 0)
            I = np.ravel(np.where(kkk == 1))
            # I = np.ravel(np.array(np.where(pi[N1:] == j)))
            I = I[choose[I] == False]
            if (I.shape[0] != 0):
                if (rho[j] == 0):
                    s = np.argmin(d[N1 + I])
                else:
                    s = np.random.randint(I.shape[0])
                choose[I[s]] = True
                rho[j] = rho[j] + 1
            else:
                zchoose[j] = False
        return choose

    # 计算距离
    def euclidean_distances(self, A, B):
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A, BT)
        # print(vecProd)
        SqA = A ** 2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        return ED

    # 评价
    def IGD(self,popfun, PF):
        distance = np.min(self.euclidean_distances(PF, popfun), 1)
        score = np.mean(distance)
        return score

    # 选择
    def envselect(self, mixpop, N, Z, Zmin):
        # 非支配排序
        mixpopfun = self.fitness(mixpop)
        frontno, maxfno = self.NDsort(mixpopfun, N)
        Next = frontno < maxfno
        # 选择最后一个front的解
        Last = np.ravel(np.array(np.where(frontno == maxfno)))
        choose = self.lastselection(mixpopfun[Next, :], mixpopfun[Last, :], N - np.sum(Next), Z, Zmin)
        Next[Last[choose]] = True
        # 生成下一代
        pop = copy.deepcopy(mixpop[Next, :])
        return pop

    # 主函数
    def excute(self):
        # 画图部分
        if (self.M <= 3):
            fig = plt.figure()
            ax = Axes3D(fig)
        ###################################################################################################################################################################
        # 产生一致性的参考点和随机初始化种群
        Z, N = self.uniformpoint(self.POP_SIZE)  # 生成一致性的参考解
        popu, theory_pof = self.initial_pop_and_theory_pof(N)  # 生成初始种群，理论POF,自变量个数
        popu_fitness = self.fitness(popu)  # 计算适应度函数值
        Zmin = np.array(np.min(popu_fitness, 0)).reshape(1, self.M)  # 求理想点
        # ax.scatter(Z[:,0],Z[:,1],Z[:,2],c='r')
        # ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='b')
        # 迭代过程
        for i in range(self.GENERATIONS):
            print("第{name}次迭代".format(name=i))
            matingpool = random.sample(range(N), N)
            off_spring_popu = self.crossover_and_mutation(popu[matingpool, :], self.t1, self.t2, self.pc, self.pm)  # 遗传算子,模拟二进制交叉和多项式变异
            off_spring_fitness = self.fitness(off_spring_popu)  # 计算适应度函数
            double_population = copy.deepcopy(np.vstack((popu, off_spring_popu)))
            Zmin = np.array(np.min(np.vstack((Zmin, off_spring_fitness)), 0)).reshape(1, self.M)  # 更新理想点
            popu = self.envselect(double_population, N, Z, Zmin)
            popu_fitness = self.fitness(popu)
            # 制图
            if (self.M <= 3):
                ax.cla()
                type1 = ax.scatter(popu_fitness[:, 0], popu_fitness[:, 1], popu_fitness[:, 2], c='g')
                plt.pause(0.00001)
        # 绘制PF
        if (self.M <= 3):
            type2 = ax.scatter(theory_pof[:, 0], theory_pof[:, 1], theory_pof[:, 2], c='r', marker='x', s=200)
            plt.legend((type1, type2), (u'Non-dominated solution', u'PF'))
        else:
            fig1 = plt.figure()
            plt.xlim([0, self.M])
            for i in range(popu_fitness.shape[0]):   #修改
                plt.plot(np.array(popu_fitness[i, :]))   #修改
        plt.show()
        # IGD
        score = self.IGD(popu_fitness, theory_pof)
        print(score)


if __name__=="__main__":
    #参数设置
    N_GENERATIONS2 = 750                                 # 迭代次数
    POP_SIZE2 = 200                                      # 种群大小
    function_name2 = 'DTLZ1'                                      # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
    fun_num2 = 8                                               # 目标个数
    t12 = 20                                             # 交叉参数t1
    t22 = 20                                             # 变异参数t2
    pc2 = 1                                              # 交叉概率
    pm2 = 1                                              # 变异概率

    # 执行NSGA3
    nsga3=NSGA3(N_GENERATIONS2, POP_SIZE2, function_name2, fun_num2, t12, t22, pc2, pm2)
    nsga3.excute()