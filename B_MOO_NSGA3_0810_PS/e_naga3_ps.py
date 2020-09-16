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
    def __init__(self,N_GENERATIONS,POP_SIZE,FUN_NUM,t1,t2,pc,pm,low,up,x_num,THROD,BEITA):
        self.GENERATIONS = N_GENERATIONS  # 迭代次数
        self.POP_SIZE = POP_SIZE  # 种群大小
        self.M = FUN_NUM  # 目标个数
        self.X_COUNT = x_num # 定义自变量个数
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
    def fitness_population(self, cipf_np, popu):
        """
        全局的适应度函数
        :param demands_np:
        :param cipf_np:
        :param popu:
        :return:
        """
        y1_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 与居民区可达性
        y2_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 与居民区公平性
        y3_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 与服务设施可达性
        y4_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 与服务设施公平性
        y5_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 覆盖人口
        y6_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 投资成本
        y7_values_double = np.empty(shape=(popu.shape[0],), dtype=np.float32)  # 等待时间
        for i in range(popu.shape[0]):
            demands_np=popu[i,:,:,:,:]
            # 开始时间
            # start_time = time.time()
            self.calculate_single_provider_gravity_value_np(demands_np)
            # 居民区 可达性适应度值，该值越大越好
            y1_values_double[i] = self.calculate_global_accessibility_numpy(demands_np)
            # 居民区  计算公平性数值，该值越小表示越公平
            y2_values_double[i] = self.calculate_global_equality_numpy(demands_np)

            # self.calculate_single_provider_gravity_value_np(cipf_np)
            # # 服务设施 可达性适应度值，该值越大越好
            # y3_values_double[i] = self.calculate_global_accessibility_numpy(cipf_np)
            # # 服务设施 计算公平性数值，该值越小表示越公平
            # y4_values_double[i] = self.calculate_global_equality_numpy(demands_np)

            # 覆盖人口
            y5_values_double[i] = self.calcuate_global_cover_people(demands_np)
            # # 投资成本
            y6_values_double[i] = self.calculate_global_cost_numpy(demands_np )
            # # 等待时间
            y7_values_double[i] = self.calculate_global_waiting_time_numpy(demands_np)
            # 结束时间
            # end_time1 = time.time()
            # print('calculate_gravity_value() Running time: %s Seconds' % (end_time1 - start_time))

        pop_fitness = np.vstack((y1_values_double, y2_values_double, y3_values_double, y4_values_double,
                                 y5_values_double, y6_values_double, y7_values_double))
        return pop_fitness

    def calculate_global_accessibility_numpy(self,demands_numpy):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        return np.nansum(demands_numpy[:, 0, 0, 1])

    def calculate_global_equality_numpy(self, demands_numpy):
        """
        计算全局公平性值
        :param demands_numpy:
        :return:
        """
        # 计算sum population count，总的需求量
        sum_population_count = np.sum(demands_numpy[:, 0, 0, 0])
        a = np.nansum(
            np.multiply(
                np.divide(
                    demands_numpy[:, 0, 0, 0], sum_population_count
                ),
                demands_numpy[:, 0, 0, 1]
            )
        )
        # 计算方差
        var = np.nansum(
            np.power(
                demands_numpy[:, 0, 0, 1] - a,
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
        np.nansum(demands_np[0, :, 2, 2])  # 获取到每一个的provider的vj值，然后求和，之所以用nan，是为了过滤掉nan的值

    def calculate_global_cost_numpy(self,demands_np):
        """
        计算全局建设成本cost
        :param demands_np:
        :param solution:
        :return:
        """
        # ？？？？？？？？？？？？？？？？？？？？
        DEMANDS_COUNT = demands_np.shape[0]
        # 执行求取每个需求点的重力值
        for i in range(DEMANDS_COUNT):
            #将小汽车充电占地面积、配套设施占地面积、行车道占地面积加和可得，j点充电站占地面积。
            #土地成本
            sj=785+60*(mj/2)
            cj2=210+35*mj
        return 0

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

    def create_providers_df_conversion_numpy_gravity(self,demands_np, DEMANDS_COUNT, PROVIDERS_COUNT):
        """
        构建以providers为主导的np数组
        :param demands_np:
        :param DEMANDS_COUNT:
        :param PROVIDERS_COUNT:
        :return:
        """
        # 以供给点为主体的dict_list_dict对象
        provider_list_dict_list = []
        for i in range(DEMANDS_COUNT):
            demand_population_count = demands_np[i, 0, 0, 0]
            for j in range(PROVIDERS_COUNT):
                D_T = demands_np[i, j, 1, 2]
                provider_transposition = self.get_provider_transportion_by_provider_id(provider_list_dict_list, j)
                if provider_transposition == None:
                    provider_transposition = {}
                    provider_transposition["provider_id"] = j
                    provider_transposition["demands"] = []
                    demand = {}
                    demand["D_T"] = D_T
                    demand["demand_id"] = i
                    demand["population"] = demand_population_count
                    provider_transposition["demands"].append(demand)
                    provider_list_dict_list.append(provider_transposition)
                else:
                    demand = {}
                    demand["D_T"] = D_T
                    demand["demand_id"] = i
                    demand["population"] = demand_population_count
                    provider_transposition["demands"].append(demand)

        providers_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 2, 1), 0.000000001)
        for j in range(PROVIDERS_COUNT):
            for i in range(DEMANDS_COUNT):
                providers_np[j, i, 0, 0] = provider_list_dict_list[j]["demands"][i]["population"]
                providers_np[j, i, 1, 0] = provider_list_dict_list[j]["demands"][i]["D_T"]
        return providers_np

    def calculate_single_provider_gravity_value_np(self, demands_np):
        """
        函数：计算重力值
        最费时的运算步骤，预计每一个solution，需要耗时0.5S
        :param demands_np:
        :param DEMANDS_COUNT:
        :return:
        """
        DEMANDS_COUNT=demands_np.shape[0]
        # 执行求取每个需求点的重力值
        for i in range(DEMANDS_COUNT):
            mask = demands_np[i, :, 1, 2] <= self.THROD  # D_T在一定有效范围内的
            gravity_value = np.sum(
                np.divide(
                    np.add(demands_np[i, :, 0, 2][mask], demands_np[i, :, 3, 2][mask]),  # 快充+慢充+solution
                    np.multiply(
                        np.power(demands_np[i, :, 1, 2][mask], self.BEITA),  # D_T
                        demands_np[i, :, 2, 2][mask]  # vj
                    )
                )
            )
            demands_np[i, :, :, 1] = gravity_value

    def calculate_provider_vj_numpy(self, demands_np):
        """
        计算vj
        :param demands_np:
        :param DEMANDS_COUNT:
        :param PROVIDERS_COUNT:
        :return:
        """
        DEMANDS_COUNT=demands_np.shape[0]   #获取需求点数量
        PROVIDERS_COUNT=demands_np.shape[1]  #获取供给者数量
        # 执行转置，计算得到每个provider的VJ
        # 以provider为一级主线，每个provider中存入所有社区的信息，以便用于vj的计算
        providers_np = self.create_providers_df_conversion_numpy_gravity(demands_np, DEMANDS_COUNT, PROVIDERS_COUNT)
        vj_list = []
        for j in range(PROVIDERS_COUNT):
            mask = providers_np[j, :, 1, 0] <= self.THROD
            vj_list.append(
                np.sum(
                    np.divide(
                        providers_np[j, :, 0, 0][mask]
                        , np.power(providers_np[j, :, 1, 0][mask], self.BEITA)
                    )
                )
            )
        # 更新vj列，每一个demand都是一样的，只是为了做一个存储。
        for i in range(DEMANDS_COUNT):
            demands_np[i, :, 2, 2] = vj_list  # vj值存放在第3行

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

    def initial_pop_demands_np(self, popsize, demands_np):
        """
        种群初始化
        :param popsize:
        :param ps:
        :return:
        """
        # 每一个供给者代表一个函数x
        # N为solution数量
        pop=np.full(shape=(popsize, demands_np.shape[0], demands_np.shape[1], demands_np.shape[2], demands_np.shape[3]), fill_value=0.0)
        for m in range(popsize):
            demands_np_copy=np.copy(demands_np)
            pop[m,:,:,:,:]=self.initial_solution(demands_np_copy)
        return pop

    def initial_solution(self, demands_np_copy):
        # ？？？？？？？？？？？？？？？？？考虑规模约束 考虑0,1
        # 获取已建CS的数量，其中quickchange的值大于0
        quick_charge = demands_np_copy[0, :, 0, 2]
        have_builded_count = len(quick_charge[np.where(quick_charge > 1)])
        # pop_have_created =np.random.randint(low=0,high=2,size=(N, have_builded_count))   #首先，区分已建和未建，针对未建的停车场生成0和1
        pop_creating = np.random.randint(low=0, high=2,size=(self.X_COUNT - have_builded_count))  # 首先，区分已建和未建，针对未建的停车场生成0和1
        creating_ps = np.random.randint(low=self.low[0,0], high=self.up[0,0] + 1, size=(self.X_COUNT - have_builded_count))  # 新建的设施
        # 将新建的设施creating_ps分布到每个位置上pop_creating
        creating_ps_2 = pop_creating * creating_ps
        # 将值填充到demands_np中
        k = 0
        for i in range(len(quick_charge)):
            if quick_charge[i] <= 1:
                quick_charge[i] = creating_ps_2[k]
                k = k + 1
        #将每一个demand体的provider赋值一遍
        for j in range(demands_np_copy.shape[0]):
            demands_np_copy[j, :, 0, 2] = quick_charge
        return demands_np_copy

    def crossover_and_mutation(self,pop, t1, t2, pc, pm):
        """
        模拟二进制交叉和多项式变异
        :param pop:
        :param t1:
        :param t2:
        :param pc:
        :param pm:
        :return:
        """
        # ？？？？？？？？？？？？？？？？？？？？
        # 考虑规模约束 考虑0,1
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

    def NDsort(self,mixpop, N, M):
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
    def envselect(self, demands_np,cipf_np,mixpop, N, Z, Zmin,  M):
        # 非支配排序
        mix_pop_fitness = self.fitness_population(demands_np, cipf_np, mixpop)
        frontno, maxfno = self.NDsort(mix_pop_fitness, N, M)
        Next = frontno < maxfno
        # 选择最后一个front的解
        Last = np.ravel(np.array(np.where(frontno == maxfno)))
        choose = self.lastselection(mix_pop_fitness[Next, :], mix_pop_fitness[Last, :], N - np.sum(Next), Z, Zmin)
        Next[Last[choose]] = True
        # 生成下一代
        pop = copy.deepcopy(mixpop[Next, :])
        return pop

    # 主函数
    def excute(self, demands_np,cipf_np):
        # 创建和计算vj，便于后面可达性计算复用，放入了demands_np array中；
        self.calculate_provider_vj_numpy(demands_np)
        self.calculate_provider_vj_numpy(cipf_np)

        # 产生一致性的参考点和随机初始化种群
        Z, popsize = self.uniformpoint(self.POP_SIZE)  # 生成一致性的参考解
        popu = self.initial_pop_demands_np(popsize, demands_np)  # 生成初始种群，修改了ps
        popu_fitness = self.fitness_population( cipf_np, popu)  # 计算适应度函数值
        Zmin = np.array(np.min(popu_fitness, 0)).reshape(1, self.M)  # 求理想点
        # 迭代过程
        for i in range(self.GENERATIONS):
            print("第{name}次迭代".format(name=i))
            matingpool = random.sample(range(popsize), popsize)
            off_spring_popu = self.crossover_and_mutation(popu[matingpool, :], self.t1, self.t2, self.pc, self.pm)  # 遗传算子,模拟二进制交叉和多项式变异
            off_spring_fitness = self.fitness_population(demands_np, cipf_np, off_spring_popu)  # 计算适应度函数
            double_popu = copy.deepcopy(np.vstack((popu, off_spring_popu)))
            Zmin = np.array(np.min(np.vstack((Zmin, off_spring_fitness)), 0)).reshape(1, self.M)  # 更新理想点
            popu = self.envselect(demands_np,cipf_np,double_popu, popsize, Z, Zmin,  self.M)
            popu_fitness = self.fitness_population(demands_np, cipf_np, popu)
        # 绘制PF
        fig1 = plt.figure()
        plt.xlim([0, self.M])
        for i in range(popu_fitness.shape[0]):  # 修改
            plt.plot(np.array(popu_fitness[i, :]))  # 修改
        plt.show()
        # IGD
        # score = self.IGD(popu_fitness, theory_pof)
        # print(score)


if __name__=="__main__":
    #参数设置  代
    N_GENERATIONS2 = 400                                 # 迭代次数
    # 区
    POP_SIZE2 = 200                                      # 种群大小
    fitness_num2 = 7                                      # 目标个数，fitness 函数的个数
    t12 = 20                                             # 交叉参数t1
    t22 = 20                                             # 变异参数t2
    pc2 = 1                                              # 交叉概率
    pm2 = 1                                              # 变异概率

    DB_NAME = "admin"     # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps" # COLLECTION 名称

    COMMUNITIES_COUNT = 184     # 需求点，即小区，个数
    # POI设施点，个数，类似于需求点，小区
    CIPF_COUNT=184
    # 供给点，即充电桩，的个数，与X_num2保持一致
    x_num2=PROVIDERS_COUNT = 40

    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 1  # 有效距离或者时间的阈值

    low2 = np.full(fill_value=3,shape=(1, x_num2))    # 最低阈值，至少建设3个
    up2 = np.full(fill_value=10, shape=(1, x_num2))  # 最高阈值，最多建设10个

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_np = mongo_operater_obj.find_records_format_in_numpy_gravity(0, COMMUNITIES_COUNT, PROVIDERS_COUNT)  #必须要先创建索引，才可以执行
    # ????????????????????????????????????  待修改
    cipf_np=mongo_operater_obj.find_records_format_in_numpy_gravity(0, CIPF_COUNT, PROVIDERS_COUNT)

    # 执行NSGA3
    nsga3=NSGA3(N_GENERATIONS2, POP_SIZE2, fitness_num2, t12, t22, pc2, pm2, low2, up2, x_num2,THROD,BEITA)
    nsga3.excute(demands_np,cipf_np)