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
import a_mongo_operater

class NSGA3:
    def __init__(self, N_GENERATIONS, POP_SIZE, FUN_NUM, t1, t2, pc, pm, low, up, old_providers_count, petential_new_providers_count, THROD, BEITA):
        self.GENERATIONS = N_GENERATIONS  # 迭代次数
        self.POP_SIZE = POP_SIZE  # 种群大小
        self.M = FUN_NUM  # 目标个数
        self.old_providers_count=old_providers_count   #已建充电站数量
        self.petential_new_providers_count = petential_new_providers_count # 定义自变量个数，需要新建的充电站数量
        self.t1 = t1  # 交叉参数t1
        self.t2 = t2  # 变异参数t2
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率
        self.low=low
        self.up=up
        self.THROD=THROD
        self.BEITA=BEITA

    ##############################################begin:fitness#########################################################

    # fitness计算
    def fitness(self, demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, popu):
        """
        全局的适应度函数
        :param demands_np:
        :param cipf_np:
        :param popu:
        :return:
        """
        actural_popsize = popu.shape[0]
        y1_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与居民区可达性
        y2_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与居民区公平性
        # y3_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与服务设施可达性
        # y4_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与服务设施公平性
        y5_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 覆盖人口
        y6_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 投资成本
        # y7_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 等待时间
        for i in range(actural_popsize):
            solution=popu[i,:]
            solution_join = np.hstack((np.full(shape=self.old_providers_count, fill_value=0.01), solution))  #把前面的已建部分的0.01补齐；
            # demands_provider_np 计算vj
            self.update_provider_vj(demands_provider_np, demands_pdd_np, solution_join)
            # cipf_provider_np 计算vj
            # self.update_provider_vj(cipf_provider_np, cipf_pdd_np, solution_join)
            # 开始时间
            # start_time = time.time()
            self.calculate_single_provider_gravity_value_np(demands_provider_np,demands_pdd_np,solution_join)
            # 居民区 可达性适应度值，该值越大越好
            y1_values_double[i] = self.calculate_global_accessibility_numpy(demands_pdd_np)
            # 居民区  计算公平性数值，该值越小表示越公平
            y2_values_double[i] = self.calculate_global_equality_numpy(demands_pdd_np)

            # self.calculate_single_provider_gravity_value_np(cipf_np)
            # # 服务设施 可达性适应度值，该值越大越好
            # y3_values_double[i] = self.calculate_global_accessibility_numpy(cipf_np)
            # # 服务设施 计算公平性数值，该值越小表示越公平
            # y4_values_double[i] = self.calculate_global_equality_numpy(cipf_np)

            # 覆盖人口，越大越好
            y5_values_double[i] = self.calcuate_global_cover_people(demands_provider_np)
            # if i>10:
            #     print("test")
            # # 投资成本，越小越好
            y6_values_double[i] = self.calculate_global_cost_numpy(solution_join)
            # # # 等待时间，越小越好
            # y7_values_double[i] = self.calculate_global_waiting_time_numpy(demands_np)
            # 结束时间
            # end_time1 = time.time()
            # print('calculate_gravity_value() Running time: %s Seconds' % (end_time1 - start_time))
        # 统一转成最小化问题
        # pop_fitness = np.vstack((10/y1_values_double, y2_values_double, y3_values_double, y4_values_double,10000000/y5_values_double, 0.0001*y6_values_double, y7_values_double)).T
        pop_fitness = np.vstack((10/y1_values_double, 1000*y2_values_double, 10000000/y5_values_double, 0.0001*y6_values_double)).T
        return pop_fitness

    def calculate_global_accessibility_numpy(self,demands_pdd_np):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        return np.nansum(demands_pdd_np[:,1])

    def calculate_global_equality_numpy(self, demands_pdd_np):
        """
        计算全局公平性值
        :param demands_numpy:
        :return:
        """
        # 计算sum population count，总的需求量
        sum_population_count = np.sum(demands_pdd_np[:, 0])
        a = np.nansum(
            np.multiply(
                np.divide(
                    demands_pdd_np[:, 0], sum_population_count
                ),
                demands_pdd_np[:, 1]
            )
        )
        # 计算方差
        var = np.nansum(
            np.power(
                demands_pdd_np[:, 1] - a,
                2
            )
        )
        return var

    def calcuate_global_cover_people(self, demands_np):
        """
        获取覆盖全局人口的总和
        :param demands_np:
        :return:
        """
        vj = demands_provider_np[:, 0, 2]
        vj[np.isinf(vj)] = 0
        vj[np.isnan(vj)] = 0
        return np.nansum(vj)  # 获取到每一个的provider的vj值，然后求和，之所以用nan，是为了过滤掉nan的值

    def calculate_global_cost_numpy(self,solution_join):
        """
        计算全局建设成本cost
        :param demands_np:
        :param solution:
        :return:
        """
        #总开销
        sum_cost=0.0
        # 执行求取每个需求点的重力值
        for i in range(solution_join.shape[0]):
            # 方式一：新建，已有设施不做调整
            if solution_join[i] !=0.01:
                # 表明新建
                #将小汽车充电占地面积、配套设施占地面积、行车道占地面积加和可得，j点充电站占地面积。
                #土地成本
                sj=785+60*(solution_join[i]/2)
                cj2=210+35*solution_join[i]
                sum_cost += sj
                sum_cost += cj2
        return sum_cost

    def calculate_global_waiting_time_numpy(self,demands_np, solution ):
        """
        计算全局等候时间
        :param demands_np:
        :param solution:
        :return:
        """
        # ？？？？？？？？？？？？？？？？？？？？
        return 0


    def get_provider_transportion_by_provider_id(self,provider_list_dict_list, provider_id):
        """
        通过provider_id获取provider
        :param provider_list_dict_list:
        :param provider_id:
        :return:
        """
        for provider in provider_list_dict_list:
            if provider["provider_id"] == provider_id:
                return provider
        return None

    def create_providers_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        构建以providers为主导的np数组
        :param demands_np:
        :param new_providers_count:
        :param solution_join:
        :return:
        """
        providers_np = np.full((demands_provider_np.shape[0], demands_provider_np.shape[1], 2), 0.000000001)
        providers_np[:, :, 0] = np.tile(demands_pdd_np[:,0],(demands_provider_np.shape[0],1)) #pdd
        providers_np[:, :, 1] = demands_provider_np[:,:,1]    #D_T
        # 将solution部分增加进去
        mask = solution_join==0
        mask=mask.reshape(mask.shape[0],1)
        mask2=np.tile(mask,(1,demands_provider_np.shape[1]))
        providers_np[:, :, 0][mask2]=0
        return providers_np

    def calculate_single_provider_gravity_value_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        函数：计算重力值
        最费时的运算步骤，预计每一个solution，需要耗时0.5S
        :param demands_np:
        :param DEMANDS_COUNT:
        :return:
        """
        DEMANDS_COUNT = demands_provider_np.shape[1]
        # 执行求取每个需求点的重力值
        for i in range(DEMANDS_COUNT):
            mask = demands_provider_np[:, i, 1] <= self.THROD  # D_T在一定有效范围内的
            gravity_value = np.nansum(
                np.divide(
                    np.add(demands_provider_np[:, i, 0][mask], solution_join[mask]),  # 快充+solution
                    np.multiply(
                        np.power(demands_provider_np[:, i, 1][mask], self.BEITA),  # D_T
                        demands_provider_np[:, i, 2][mask]  # vj
                    )
                )
            )
            demands_pdd_np[i, 1] = gravity_value

    def update_provider_vj(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        计算vj
        :param demands_np:
        :param DEMANDS_COUNT:
        :param PROVIDERS_COUNT:
        :return:
        """
        # 执行转置，计算得到每个provider的VJ
        # 以provider为一级主线，每个provider中存入所有社区的信息，以便用于vj的计算
        # 增加一个solution的过滤，因为新增部分有些建，有些不建
        effective_providers_np = self.create_providers_np(demands_provider_np, demands_pdd_np, solution_join)
        vj_list = []
        for j in range(effective_providers_np.shape[0]):
            mask = effective_providers_np[j, :, 1] <= self.THROD  # 对交通时间做对比
            vj_list.append(
                np.sum(
                    np.divide(
                        effective_providers_np[j, :, 0][mask]
                        , np.power(effective_providers_np[j, :, 1][mask], self.BEITA)
                    )
                )
            )
        # 更新vj列
        vj_np = np.array(vj_list).reshape(len(vj_list), 1)
        demands_provider_np[:, :, 2] = np.tile(vj_np, (1, demands_provider_np.shape[1]))
        # print("test")

    ##############################################end:fitness#########################################################


    def uniformpoint(self, popsize):
        """
        构建参考点
        :param popsize:
        :return:
        """
        H1 = 1
        while (comb(H1 + self.M - 1, self.M - 1) <= popsize):
            H1 = H1 + 1
        H1 = H1 - 1
        Z = np.array(list(combinations(range(H1 + self.M - 1), self.M - 1))) - np.tile(
            np.array(list(range(self.M - 1))), (int(comb(H1 + self.M - 1, self.M - 1)), 1))
        Z = (np.hstack((Z, H1 + np.zeros((Z.shape[0], 1)))) - np.hstack((np.zeros((Z.shape[0], 1)), Z))) / H1
        if H1 < self.M:
            H2 = 0
            while (comb(H1 + self.M - 1, self.M - 1) + comb(H2 + self.M - 1, self.M - 1) <= popsize):
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
        popsize = Z.shape[0]
        return Z, popsize

    def initial_population(self, popsize):
        """
        种群初始化
        :param popsize:
        :param ps:
        :return:
        """
        # 每一个供给者代表一个函数x
        # N为solution数量
        pop = np.full(shape=(popsize, self.petential_new_providers_count), fill_value=0)
        for m in range(popsize):
            # 方式一：新增方式，已建不调整；方式二：新建，已建可调整；方式三：总数约束，新建
            # 以下只针对未建服务设施的站点，部分新增设施，已建服务设施的站点，不做考虑
            creating_mark = np.random.randint(low=0, high=2,
                                              size=(self.petential_new_providers_count))  # 首先，区分已建和未建，针对未建的停车场生成0和1
            creating_ps = np.random.randint(low=self.low, high=self.up + 1,
                                            size=(self.petential_new_providers_count))  # 新建的设施
            pop[m, :] = creating_mark * creating_ps
        return pop

    # Function to carry out the crossover
    def crossover(self, solution_a, solution_b, pc):
        crossover_prob = random.random()
        if crossover_prob > pc:
            return np.hstack((solution_a[0:20], solution_b[20:]))
        else:
            return solution_a

    # Function to carry out the mutation operator
    def mutation(self, solution, pm):
        mutation_prob = random.random()
        if mutation_prob < pm:
            creating_mark = np.random.randint(low=0, high=2,
                                              size=(solution.shape[0]))  # 首先，区分已建和未建，针对未建的停车场生成0和1
            solution = np.random.randint(low=self.low, high=self.up + 1,
                                            size=(solution.shape[0]))  # 新建的设施
            solution=creating_mark*solution
        return solution

    def crossover_and_mutation_simple(self, popu, pc, pm):
        off_spring=np.full(shape=(popu.shape[0], self.petential_new_providers_count), fill_value=0)
        for i in range(popu.shape[0]):
            a1 = random.randint(0, popu.shape[0]-1)
            b1 = random.randint(0, popu.shape[0]-1)
            # 通过crossover和mutation的方式生成新的个体
            solution=self.crossover(popu[a1,:], popu[b1,:],pc)
            off_spring[i,:]=self.mutation(solution,pm)
        return off_spring

    def crossover_and_mutation(self, popu, t1, t2, pc, pm):
        """
        模拟二进制交叉和多项式变异
        :param popu:
        :param t1:
        :param t2:
        :param pc:
        :param pm:
        :return:
        """
        # ？？？？？？？？？？？？？？？？？？？？
        # 考虑规模约束 考虑0,1
        pop1 = copy.deepcopy(popu[0:int(popu.shape[0] / 2), :])
        pop2 = copy.deepcopy(popu[(int(popu.shape[0] / 2)):(int(popu.shape[0] / 2) * 2), :])
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
        low = np.full(shape=(2 * N, D),fill_value=self.low)
        up = np.full(shape=(2 * N, D),fill_value=self.up)
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
        creating_mark = np.random.randint(low=0, high=2,
                                          size=(2 * N, D))  # 首先，区分已建和未建，针对未建的停车场生成0和1
        return off*creating_mark

    def NDsort(self,mixpop, N):
        """
        非支配排序
        :param mixpop:
        :param N:
        :param M:
        :return:
        """
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

    def pdist(self,x, y):
        """
        求两个向量矩阵的余弦值,x的列数等于y的列数
        :param x:
        :param y:
        :return:
        """
        x0 = x.shape[0]
        y0 = y.shape[0]
        xmy = np.dot(x, y.T)  # x乘以y
        xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)
        ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
        xmmym = np.dot(xm, ym)
        cos = xmy / xmmym
        return cos

    def lastselection(self,popfun1, popfun2, K, Z, Zmin):
        """
        临界面上进行选择
        :param popfun1:
        :param popfun2:
        :param K:
        :param Z:
        :param Zmin:
        :return:
        """
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
            # temp = np.linalg.inv(np.mat(popfun[extreme, :]))  # 逆矩阵
            temp = np.linalg.pinv(np.mat(popfun[extreme,:]))   #广义逆矩阵 原来是求广义逆矩阵的，所以导致多个目标的时候可以求到广义逆矩阵。
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

    def euclidean_distances(self, A, B):
        """
        计算距离
        :param A:
        :param B:
        :return:
        """
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
    def envselect(self, demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, mixpop, N, Z, Zmin):
        # 非支配排序
        mix_pop_fitness = self.fitness(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, mixpop)
        frontno, maxfno = self.NDsort(mix_pop_fitness, N)
        Next = frontno < maxfno
        # 选择最后一个front的解
        Last = np.ravel(np.array(np.where(frontno == maxfno)))
        choose = self.lastselection(mix_pop_fitness[Next, :], mix_pop_fitness[Last, :], N - np.sum(Next), Z, Zmin)
        Next[Last[choose]] = True
        # 生成下一代
        pop = copy.deepcopy(mixpop[Next, :])
        return pop

    # 主函数
    def excute(self, demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np):
        # 产生一致性的参考点
        Z, actural_popsize = self.uniformpoint(self.POP_SIZE)  # 生成一致性的参考解
        # 随机初始化种群
        popu = self.initial_population(actural_popsize)
        popu_fitness = self.fitness(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, popu)  # 计算适应度函数值
        Zmin = np.array(np.min(popu_fitness,0)).reshape(1, self.M)  # 求理想点
        # 迭代过程
        for i in range(self.GENERATIONS):
            print("第{name}次迭代".format(name=i))
            matingpool = random.sample(range(actural_popsize), actural_popsize)
            # off_spring_popu = self.crossover_and_mutation(popu[matingpool,:], self.t1, self.t2, self.pc, self.pm)# 遗传算子,模拟二进制交叉和多项式变异
            off_spring_popu = self.crossover_and_mutation_simple(popu[matingpool,:],  self.pc, self.pm)# 遗传算子,模拟二进制交叉和多项式变异
            off_spring_fitness = self.fitness(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, off_spring_popu)  # 计算适应度函数
            double_popu = copy.deepcopy(np.vstack((popu, off_spring_popu)))
            Zmin = np.array(np.min(np.vstack((Zmin, off_spring_fitness)), 0)).reshape(1, self.M)  # 更新理想点
            popu = self.envselect(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np, double_popu,actural_popsize, Z, Zmin)
            popu_fitness = self.fitness(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np,  popu)
        # 绘制PF
        fig1 = plt.figure()
        plt.xlim([0, self.M])
        for i in range(popu_fitness.shape[0]):
            plt.plot(np.array(popu_fitness[i, :]))
        plt.show()
        # IGD
        # score = self.IGD(popu_fitness, theory_pof)
        # print(score)


if __name__=="__main__":
    #参数设置  代
    N_GENERATIONS2 = 400                                # 迭代次数
    # 区
    POP_SIZE2 = 200                                      # 种群大小
    fitness_num2 = 4                                      # 目标个数，fitness 函数的个数
    t12 = 20                                             # 交叉参数t1
    t22 = 20                                             # 变异参数t2
    pc2 = 0.6                                              # 交叉概率
    pm2 = 0.05                                              # 变异概率



    DEMANDS_COUNT = 184     # 需求点，即小区，个数
    # POI设施点，个数，类似于需求点，小区
    CIPF_COUNT=184
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 49
    OLD_PROVIDERS_COUNT = 40

    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 1  # 有效距离或者时间的阈值

    low2 = 3# 最低阈值，至少建设3个
    up2 = 10 # 最高阈值，最多建设10个

    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps"  # COLLECTION 名称
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np = mongo_operater_obj.find_records_format_numpy_2(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  #必须要先创建索引，才可以执行
    # ????????????????????????????????????  待修改
    cipf_provider_np, cipf_pdd_np=mongo_operater_obj.find_records_format_numpy_2(0, CIPF_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)

    # 执行NSGA3
    nsga3=NSGA3(N_GENERATIONS2, POP_SIZE2, fitness_num2, t12, t22, pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT,PENTENTIAL_NEW_PROVIDERS_COUNT,THROD,BEITA)
    nsga3.excute(demands_provider_np, demands_pdd_np,cipf_provider_np, cipf_pdd_np)