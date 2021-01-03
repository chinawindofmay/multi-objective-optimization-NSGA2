"""
参考地址：https://blog.csdn.net/qq_36449201/article/details/81046586
作者：华电小炸扎
主要的修改点在于：
（1）改了非支配排序的算法；
（2）改了遗传算子的部分，使之符合site location 的需求；
（3）改了目标函数评价的部分，使之符合site location的需求；
（4）改了可视化与评价的部分，使之符合site location的需求
"""

import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib
import a_mongo_operater_PS_JY_TIME
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import seaborn as sns
import pandas as pd
import copy
from itertools import combinations
from scipy.special import comb
sns.set(style='whitegrid', context='notebook')
# sns.set(style="ticks", color_codes=True)


class NSGA3():
    ##############################################begin:fitness#########################################################
    # <editor-fold desc="fitness">
    def fitness(self, demands_provider_np, demands_pdd_np, popu):
        """
        目标函数1：行驶时间总和最小；
        目标函数2：Waiting time 最小；
        目标函数3：公平性最大；
        目标函数4：覆盖人口最大；
        目标函数5：建设成本最低；
        # fitness计算，全局的适应度函数
        :param demands_np:
        :param popu:
        :return:
        """
        actural_popsize = popu.shape[0]
        y1_driving_time = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 行驶时间
        y2_waiting_time = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # Waiting time 最小
        y3_inequlity = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 与居民区公平性
        y4_cover_people = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 覆盖人口
        y5_cost = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 建设成本
        for i in range(actural_popsize):
            solution = popu[i, :]
            solution_join = np.hstack((np.full(shape=self.old_providers_count, fill_value=self.SMALL_INF), solution))  # 把前面的已建部分的0.01补齐；
            demands_provider_np_temp = copy.deepcopy(demands_provider_np)
            demands_pdd_np_temp = copy.deepcopy(demands_pdd_np)
            # 目标函数1：行驶时间总和最小；目标函数2：Waiting time 最小；
            y2_waiting_time[i],y1_driving_time[i]=self.driving_time_waiting_time(demands_provider_np_temp, demands_pdd_np_temp,solution_join)
            # demands_provider_np 计算vj
            self.update_provider_vj(demands_provider_np_temp, demands_pdd_np_temp, solution_join)
            self.calculate_single_provider_gravity_value_np(demands_provider_np_temp, demands_pdd_np_temp, solution_join)
            # 居民区 可达性
            # TESTCODE 测试代码 测试可达性计算是不是对的   Test value 值应该和provider的总和一致，如4*10
            # self.test_global_accessibility_numpy(demands_pdd_np_temp)
            # 最小化：居民区  计算不公平性数值，该值越小，差异性越小，表示越公平
            y3_inequlity[i] = self.calculate_global_inequality_np(demands_pdd_np_temp)
            # # 最小化，已经将覆盖人口越大越好转为最小化问题
            y4_cover_people[i] = self.calcuate_global_cover_people(demands_provider_np_temp,solution_join)
            # 建设成本，越小越好
            y5_cost[i]=self.calculate_global_cost_numpy(solution_join)
        # 统一转成最小化问题
        # pop_fitness = np.vstack((1e-3*y1_driving_time-np.min(1e-3*y1_driving_time),
        #                          y2_waiting_time-np.min(y2_waiting_time),
        #                          1e-2 * y3_inequlity-np.min(1e-2 * y3_inequlity),
        #                          1e5*y4_cover_people-np.min(1e5*y4_cover_people),
        #                          1e-4*y5_cost-np.min(1e-4*y5_cost))).T

        pop_fitness = np.vstack((
                                    (y1_driving_time - np.min(y1_driving_time))/(np.max(y1_driving_time)-np.min(y1_driving_time)),
                                    (y2_waiting_time - np.min(y2_waiting_time))/(np.max(y2_waiting_time)-np.min(y2_waiting_time)),
                                    (y3_inequlity - np.min(y3_inequlity))/(np.max(y3_inequlity)-np.min(y3_inequlity)),
                                    (y4_cover_people - np.min(y4_cover_people))/(np.max(y4_cover_people)-np.min(y4_cover_people)),
                                    (y5_cost - np.min(y5_cost))/(np.max(y5_cost)-np.min(y5_cost))
        )).T
        # pop_fitness = np.vstack((1e-5*y1_driving_time, 1e-2 * y3_inequlity, 1e-4*y5_cost)).T
        return pop_fitness


    def driving_time_waiting_time(self, demands_provider_np, demands_pdd_np,solution_join):
        """
        目标函数1：行驶时间总和最小；
        目标函数2：Waiting time 最小；
        :param demands_provider_np:
        :param demands_pdd_np:
        :return:
        """
        waiting_time_array=np.full(shape=(4,), fill_value=0.0, dtype=np.float32)
        driving_time_array=np.full(shape=(4,), fill_value=0.0, dtype=np.float32)
        for k in range(0, 4):
            # 需求量 W_I
            W_I=np.reshape(demands_pdd_np[:,k],(1,len(demands_pdd_np[:,0])))
            #筛选出符合的provider（新建+已建）
            mask= solution_join >= self.SMALL_INF
            # 充电桩供给量 M_J:久量+新量，分别取出两截，拼起来，然后再做筛选
            M_J=np.hstack((demands_provider_np[:,0,0][0:self.old_providers_count],solution_join[self.old_providers_count:]))[mask]
            M_J=np.reshape(M_J,(1,len(M_J)))
            # TRAVELTIME_IJ 行是demand，列是provider
            TRAVELTIME_IJ=demands_provider_np[:,:,1+k].T
            TRAVELTIME_IJ=TRAVELTIME_IJ[:,mask]
            # 选择权重
            Y_IJ_08 = self.calculate_Y_IJ(TRAVELTIME_IJ, W_I)
            # 计算等候时间
            waiting_time_array[k] = self.waiting_time_one_moment_new(W_I, M_J, Y_IJ_08)
            # 计算驾车时间
            driving_time_array[k] = self.driving_time_one_moment(W_I, TRAVELTIME_IJ, Y_IJ_08)
        return np.sum(waiting_time_array),np.sum(driving_time_array)


    def waiting_time_one_moment_new(self,W_I, M_J,  Y_IJ):
        tc=6           # 服务时间，24小时提供服务；
        tf=2           #每辆车充电时长；
        # 第一步：计算用户达到率
        Tao_J=np.dot(W_I,Y_IJ/tc)

        # 第二步：单位时间内充电站平均服务能力
        U_J=M_J/tf

        # 第三步：充电站排队系统服务强度：，由于ROU_J会存在大于1的情况，从而会促使后面求解
        ROU_J=Tao_J/U_J
        # 加一个判断
        if np.any(ROU_J/M_J>1):
            # 当ROU_J/M_J<1条件下才能得到，即要求顾客的平均到达率小于系统的平均服务率，才能使系统达到统计平衡。
            T2=self.BIG_INF
            return T2

        # 第四步：充电站内充电桩全部空闲概率：
        P_J=np.full(shape=(1,M_J.shape[1]),fill_value=0.0)
        for j in range(M_J.shape[1]):######修改
            temp = 0
            for k in range(M_J.shape[1]):#修改
                temp+=(np.power(ROU_J[0,j],k))/(np.math.factorial(k))
            p_j0=1/(temp+(np.power(ROU_J[0,j],M_J[0,j]))/(np.math.factorial(M_J[0,j])*(1-ROU_J[0,j]/M_J[0,j])))  ##修改
            P_J[0,j]=p_j0

        # 第五步：计算排队等候时间期望
        W_Jq=np.full(shape=(1,M_J.shape[1]),fill_value=0.0)
        for j in range(M_J.shape[1]):
            w_jq=(np.power(ROU_J[0,j],M_J[0,j]+1)*P_J[0,j])/(M_J[0,j]*Tao_J[0,j]*np.math.factorial(M_J[0,j])*np.power(1-ROU_J[0,j]/M_J[0,j],2))#修改
            W_Jq[0,j]=w_jq

        # 第六步：所有用户的总的等待花费时间
        T2=0
        for j in range(M_J.shape[1]):
            T2+=W_Jq[0,j]*Tao_J[0,j]*tc
        if T2<0:
            T2=self.BIG_INF

        return T2

    def calculate_Y_IJ(self,TRAVELTIME_IJ,W_I):
        # 出行时间的阻尼函数，衰减函数
        F_DIJ = 1 / TRAVELTIME_IJ
        Sum_Dij_I = np.sum(F_DIJ, axis=1)
        # 计算选择权重
        Y_IJ = np.full(shape=(TRAVELTIME_IJ.shape), fill_value=0.0)
        for i in range(W_I.shape[1]):
            Y_IJ[i, :] = F_DIJ[i, :] / Sum_Dij_I[i]
        return  Y_IJ

    # 目标函数1：行驶时间总和最小；
    def driving_time_one_moment(self,W_I,  TRAVELTIME_IJ, Y_IJ):
        t1=np.dot(W_I, Y_IJ)
        t1=t1.reshape(t1.shape[1],)
        t1[np.argwhere(np.isnan(t1))]=self.SMALL_INF
        # return np.nansum(np.dot(np.dot(W_I, Y_IJ), TRAVELTIME_IJ.T))
        return np.nansum(np.dot(t1, TRAVELTIME_IJ.T))

    def calculate_global_cost_numpy(self, solution):
        """
        目标函数4：计算全局建设成本cost
        :param demands_np:
        :param solution:
        :return:
        """
        mark = solution > self.SMALL_INF   # 新建设施
        #将小汽车充电占地面积、配套设施占地面积、行车道占地面积加和可得，j点充电站占地面积。
        #土地成本
        sj=785*mark+60*(solution / 2)
        cj2= 210 * mark + 35 * solution
        sum_cost = np.sum(sj)+np.sum(cj2)
        return sum_cost

    def test_global_accessibility_numpy(self, demands_pdd_np):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 适应度值，该值越大越好
        # 空间可达性数值实则代表相应研究单元人均拥有的公共服务资源数量，如每人医院床位数、每人教师数等，是一个人均的概念，入错此时再将其乘以人口，则得到了provider的总量。
        # 如果对其直接加和，是没有实际意义的，因为是一个均值单位的概念。
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        # 求取其均值，如果均值上来了，我们认为也是符合的

        print("理论值：",self.E_OLD_SUM+self.E_NEW_SUM,"计算得值：", np.nansum(demands_pdd_np[:, 0]*demands_pdd_np[:, 4])
                                                                      , np.nansum(demands_pdd_np[:, 1]*demands_pdd_np[:, 5])
                                                                                  , np.nansum(demands_pdd_np[:, 2]*demands_pdd_np[:, 6])
                                                                                              , np.nansum(demands_pdd_np[:, 3]*demands_pdd_np[:, 7]))

    def calculate_global_inequality_np(self, demands_pdd_np):
        """
        计算全局公平性值  可达性差异最小化为目标，通过除以一下，得到：实现设施布局的公平最大化问题
        :param demands_numpy:
        :return:
        """
        # 不同时刻的总人口
        pop_08_sum = np.nansum(demands_pdd_np[:, 0])
        pop_13_sum = np.nansum(demands_pdd_np[:, 1])
        pop_18_sum = np.nansum(demands_pdd_np[:, 2])
        pop_22_sum = np.nansum(demands_pdd_np[:, 3])
        SUM_CHARGE=self.E_NEW_SUM+self.E_OLD_SUM
        # 可达性差异最小化为目标
        return (np.nansum((demands_pdd_np[:, 4] - SUM_CHARGE / pop_08_sum) ** 2) +
                   np.nansum((demands_pdd_np[:, 5] - SUM_CHARGE / pop_13_sum) ** 2) +
                   np.nansum((demands_pdd_np[:, 6] - SUM_CHARGE / pop_18_sum) ** 2) +
                   np.nansum((demands_pdd_np[:, 7] - SUM_CHARGE / pop_22_sum) ** 2))

    def calcuate_global_cover_people(self, demands_provider_np,solution_join):
        """
        获取覆盖全局人口的总和
        :param demands_np:
        :return:
        """
        mask=solution_join>self.SMALL_INF
        vj = demands_provider_np[mask, 0, 5:]
        vj[np.isinf(vj)] = 0
        vj[np.isnan(vj)] = 0
        # 获取到每一个的provider的vj值，然后求和，之所以用nan，是为了过滤掉nan的值
        # 将最大化问题转为最小化问题
        return 1/np.nansum(vj)

    def create_providers_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        构建以providers为主导的np数组
        :param demands_np:
        :param new_providers_count:
        :param solution_join:
        :return:
        """
        providers_np = np.full((demands_provider_np.shape[0], demands_provider_np.shape[1], 2*4), 0.000000001)
        providers_np[:, :, 0] = np.tile(demands_pdd_np[:, 0], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 1] = np.tile(demands_pdd_np[:, 1], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 2] = np.tile(demands_pdd_np[:, 2], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 3] = np.tile(demands_pdd_np[:, 3], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 4] = demands_provider_np[:, :, 1]  # D_T
        providers_np[:, :, 5] = demands_provider_np[:, :, 2]  # D_T
        providers_np[:, :, 6] = demands_provider_np[:, :, 3 ]  # D_T
        providers_np[:, :, 7] = demands_provider_np[:, :, 4 ]  # D_T
        # 将solution部分增加进去
        mask = solution_join == 0
        mask = mask.reshape(mask.shape[0], 1)
        mask2 = np.tile(mask, (1, demands_provider_np.shape[1]))
        providers_np[:, :, 0][mask2] = 0
        providers_np[:, :, 1][mask2] = 0
        providers_np[:, :, 2][mask2] = 0
        providers_np[:, :, 3][mask2] = 0
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
        for k in range(0, 4):
            # 执行求取每个需求点的重力值
            for i in range(DEMANDS_COUNT):
                mask = demands_provider_np[:, i, 1+k] <= self.THROD  # D_T在一定有效范围内的
                gravity_value = np.nansum(
                    np.divide(
                        np.add(demands_provider_np[:, i, 0][mask], solution_join[mask]),  # 快充+solution
                        np.multiply(
                            np.power(demands_provider_np[:, i, 1+k][mask], self.BEITA),  # D_T
                            demands_provider_np[:, i, 5+k][mask]  # vj
                        )
                    )
                )
                demands_pdd_np[i, 4+k] = gravity_value

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
        # 因为有四个时刻，所以需要做四次循环
        for k in range(0, 4):
            vj_list = []
            for j in range(effective_providers_np.shape[0]):
                mask = effective_providers_np[j, :, 4+k] <= self.THROD  # 对交通时间做对比
                vj_list.append(
                    np.sum(
                        np.divide(
                            effective_providers_np[j, :, 0+k][mask]   #人口
                            , np.power(effective_providers_np[j, :, 4+k][mask], self.BEITA)   #时间
                        )
                    )
                )
            # 更新vj列
            vj_np = np.array(vj_list).reshape(len(vj_list), 1)
            demands_provider_np[:, :, 5+k] = np.tile(vj_np, (1, demands_provider_np.shape[1]))

    # </editor-fold>
    ##############################################end:fitness#########################################################

    ##############################################begin：population#########################################################
    # <editor-fold desc="population crossover mutation">
    def initial_population(self,popsize):
        """
        初始化种群：
        方式1：定个数is_new_building_providers_count_fixed,定规模 is_every_new_building_providers_scale_equal、
        :return:
        """
        population = np.full(shape=(popsize, self.PENTENTIAL_NP_C), fill_value=0, dtype=np.int)
        for m in range(popsize):
            if self.B_NP_CF ==True:
                #count fixed
                if self.B_NP_SF==True:
                    # count fixed，scale fixed
                    population[m, :] = self.create_solution_CF_SF(self.PENTENTIAL_NP_C, self.NP_C, self.NP_S)
                else:
                    # count fixed，scale unfixed
                    population[m, :] = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.PENTENTIAL_NP_C, E_setting=self.E_NEW_SUM, k=self.NP_C)
            elif self.B_NP_CF ==False:
                #非固定的provider count
                if self.B_NP_SF:
                    #count unfixed ，scale fixed 其本质上就是CF_SF，因为我们的总数量是规定的
                    population[m, :] = self.create_solution_CF_SF(self.PENTENTIAL_NP_C, int(self.E_NEW_SUM / self.NP_S), self.NP_S)
                else:
                    # count unfixed ，scale unfixed
                    #首先，随机明确新建 count，然后转换为count fixed，scale unfixed
                    np_count = np.random.randint(low=int(self.E_NEW_SUM / self.up), high=int(self.E_NEW_SUM / self.low), size=1)[0]
                    population[m, :]=self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.PENTENTIAL_NP_C, E_setting=self.E_NEW_SUM, k=np_count)
        return population

    def create_solution_CF_SF(self, x_dim, k, scale):
        """
         count fixed, scale fixed
        :param x_dim:
        :param k:
        :param scale:
        :return:
        """
        # 无需传入up/low/E_setting，因为用不上
        choic_index_list = np.random.choice(a=[i for i in range(x_dim)],
                                            size=k, replace=False)
        creating_mark = np.full(shape=(x_dim), fill_value=0)
        creating_mark[choic_index_list] = 1
        solution = creating_mark * scale  # 规模都是10
        return solution

    def create_solution_CF_SUF(self, lower, up, x_dim, E_setting, k):
        """
        count fixed，scale unfixed
        :param lower:
        :param up:
        :param x_dim:
        :param E_setting:
        :param k:
        :return:
        """
        # 选择出k个盘子
        choic_index_list = np.random.choice(a=[i for i in range(x_dim)], size=k,replace=False)
        creating_mark = np.full(shape=(x_dim), fill_value=0)
        creating_mark[choic_index_list] = 1
        solution = np.random.randint(low=lower, high=up + 1, size=(x_dim))  # 新建的设施
        solution = creating_mark * solution
        #求平
        return self.balance_E_setting(E_setting, choic_index_list, lower, solution, up)

    def balance_E_setting(self, E_setting, choic_index_list, low, solution, up):
        """
        将不平整的部分平整输出
        :param E_setting:
        :param choic_index_list:
        :param low:
        :param solution:
        :param up:
        :return:
        """
        solution = (solution * (E_setting / np.sum(solution))).astype(np.int)  # 规模介于up和low之间
        # 会存在部分因最大值整数约束，而超限的数值，需要先处理到【low,up】范围内
        while np.sum(solution[np.where((solution < low) & (solution != 0))]) > 0 or np.sum(solution[solution > up]) > 0:
            solution[np.where((solution < low) & (solution != 0))] += 1
            solution[solution > up] -= 1
        # 补齐因最大值整数约束导致的部分值丢失
        # 该方法对于极限情况，如：30,100，会存在多次循环的问题，所以增加一个判别条件，让其不至于循环很久，提高效率，但是这种方法有极小的可能性会导致值不准确，目前忽略未用
        iteation_count = 0
        while np.sum(solution) != E_setting:
            delate = np.sum(solution) - E_setting
            if delate > 0:
                adjust_mark = -1
            else:
                adjust_mark = +1
            for adjust_i in range(np.abs(delate)):
                adjust_index = np.random.choice(a=choic_index_list, size=1, replace=False)
                if solution[adjust_index] > low and solution[adjust_index] < up:
                    solution[adjust_index] += adjust_mark
                if delate > 0 and solution[adjust_index] == up:
                    solution[adjust_index] += adjust_mark
                if delate < 0 and solution[adjust_index] == low:
                    solution[adjust_index] += adjust_mark
            # if iteation_count<100:
            #     iteation_count+=1
            # else:
            #     break
        return solution

    def cross_and_mutation(self, population):
        """
        选择、交叉、变异，生成子代种群：
        方式1：定个数is_new_building_providers_count_fixed,定规模 is_every_new_building_providers_scale_equal、
        :return:
        """
        off_spring = np.full(shape=(population.shape[0], self.PENTENTIAL_NP_C), fill_value=0)
        for i in range(population.shape[0]):
            a1 = random.randint(0, population.shape[0] - 1)
            b1 = random.randint(0, population.shape[0] - 1)
            # count fixed
            if self.B_NP_CF==True:
                if self.B_NP_SF==True:
                    # count fixed，scale fixed
                    solution = self.cross_CF_SF(population[a1, :], population[b1, :])
                    off_spring[i, :] = self.mutation_CF_SF(solution)
                else:
                    # count fixed，scale unfixed
                    solution = self.cross_CF_SUF(a1, b1, population)
                    off_spring[i, :] = self.mutation_CF_SUF(solution)
            else:
                #  count unfixed
                if self.B_NP_SF==True:
                    # count unfixed，scale fixed
                    solution = self.cross_CUF_SF(population[a1, :], population[b1, :])
                    off_spring[i, :] = self.mutation_CUF_SF(solution)
                else:
                    # count fixed，scale unfixed
                    solution = self.cross_CUF_SUF(population[a1, :], population[b1, :])
                    off_spring[i, :] = self.mutation_CUF_SUF(solution)
        return off_spring

    def sample_index_from_effective(self, solution, count=1):
        """
         #通过抽样的方式获取到Index
        :param solution:
        :param count:
        :return:
        """
        effective_index=np.argwhere(solution >0).flatten()
        # 测试代码，如果找到了没有有效provider count的solution，则报错
        if effective_index.shape[0]==0:
            print("没有有效provider count的solution")
        # # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
        # solution_a_index_1 = np.random.randint(low=0, high=effective_index.shape[0],size=int(count))
        # return effective_index[solution_a_index_1]
        return np.random.choice(a=effective_index, size=int(count), replace=False)

    def cross_CF_SF(self, solution_a, solution_b):
        """
        交叉算子
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        crossover_prob = random.random()
        if crossover_prob < self.pc:
            #取出有值的位置index list
            count=int(self.NP_C / 2)
            solution_a_index =self.sample_index_from_effective(solution_a, count=count)  #通过抽样的方式获取到Index
            solution_b_index = self.sample_index_from_effective(solution_b, count=count)
            c=(solution_a_index == solution_b_index)
            if c.all()==False:  #表示取出来的，不是完全相等
                for item in range(count):
                    item_b=solution_b_index[item]
                    item_a=solution_a_index[item]
                    if solution_a[item_b]==0:
                        solution_a[item_b]=solution_b[item_b]
                        solution_a[item_a]= 0
        return solution_a

    def cross_CF_SUF(self, a1, b1, population):
        solution = self.cross_CF_SF(population[a1, :], population[b1, :])
        # 求平
        if np.sum(solution) != self.E_NEW_SUM:
            choic_index_list = np.argwhere(solution > 0).flatten()
            solution = self.balance_E_setting(self.E_NEW_SUM, choic_index_list, self.low, solution, self.up)
        return solution

    def cross_CUF_SF(self, solution_a, solution_b):
        """
        交叉算子
        数量不一致，规模一致
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        crossover_prob = random.random()
        if crossover_prob < self.pc:
            #取出有值的位置index list
            count_a=np.argwhere(solution_a>0).flatten().shape[0]
            count_b=np.argwhere(solution_b>0).flatten().shape[0]
            count= count_a if count_a < count_b else count_b   #取小的值
            count=int(count/2) if int(count/2)>0 else 1
            solution_a_index_1 =self.sample_index_from_effective(solution_a, count=count)  #通过抽样的方式获取到Index
            solution_b_index_1 = self.sample_index_from_effective(solution_b, count=count)
            c=(solution_a_index_1 == solution_b_index_1)
            if c.all()==False:  #表示取出来的，不是完全相等
                for item in range(count):
                    item_b=solution_b_index_1[item]
                    item_a=solution_a_index_1[item]
                    if solution_a[item_b]==0:
                        solution_a[item_b]=solution_b[item_b]
                        solution_a[item_a]= 0
        return solution_a

    def cross_CUF_SUF(self, solution_a, solution_b):
        """
        交叉算子
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        # 先按照CUF_SF的方式做交叉，然后调平
        solution = self.cross_CUF_SF(solution_a, solution_b)
        # 调平
        if np.sum(solution) != self.E_NEW_SUM:
            choic_index_list = np.argwhere(solution > 0).flatten()
            solution = self.balance_E_setting(self.E_NEW_SUM, choic_index_list, self.low, solution, self.up)
        return  solution

    def mutation_CF_SF(self, solution):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < self.pm:
            solution = self.create_solution_CF_SF(self.PENTENTIAL_NP_C, self.NP_C, self.NP_S)
        return solution

    def mutation_CF_SUF(self, solution):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < self.pm:
            # 固定的provider count，非固定的provider规模
            solution = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.PENTENTIAL_NP_C, E_setting=self.E_NEW_SUM, k=self.NP_C)
        return solution

    def mutation_CUF_SF(self, solution):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < self.pm:
            # 非固定的provider count，固定的provider 规模
            solution = self.create_solution_CF_SF(self.PENTENTIAL_NP_C, int(self.E_NEW_SUM / self.NP_S), self.NP_S)
        return solution

    def mutation_CUF_SUF(self, solution):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < self.pm:
            # 非固定的provider count，非固定的provider规模
            # 首先明确新建多少个provider count
            k = np.random.randint(low=int(self.E_NEW_SUM / self.up), high=int(self.E_NEW_SUM / self.low), size=1)[0]
            solution = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.PENTENTIAL_NP_C, E_setting=self.E_NEW_SUM, k=k)
        return solution

    # </editor-fold>
    ##############################################end：population#########################################################

    # <editor-fold desc="非支配排序、拥挤度算子、选择">
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
        off = np.vstack(
            ((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))
        # 多项式变异
        low = np.full(shape=(2 * N, D), fill_value=self.low)
        up = np.full(shape=(2 * N, D), fill_value=self.up)
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
        return off * creating_mark

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

    # 选择
    def envselect(self, demands_provider_np, demands_pdd_np, mixpop, N, Z, Zmin):
        # 非支配排序
        mix_pop_fitness = self.fitness(demands_provider_np, demands_pdd_np, mixpop)
        frontno, maxfno = self.NDsort(mix_pop_fitness, N)
        Next = frontno < maxfno
        # 选择最后一个front的解
        Last = np.ravel(np.array(np.where(frontno == maxfno)))
        choose = self.lastselection(mix_pop_fitness[Next, :], mix_pop_fitness[Last, :], N - np.sum(Next), Z, Zmin)
        Next[Last[choose]] = True
        # 生成下一代
        pop = copy.deepcopy(mixpop[Next, :])
        return pop

    # </editor-fold>


    def __init__(self, pc, pm, low, up, old_providers_count, THROD, BEITA, PENTENTIAL_NP_C, pop_size, max_iter, f_num,is_builded_providers_adjust, B_NP_CF, NP_C,B_NP_SF,NP_S, E_NEW_SUM,SMALL_INF,BIG_INF):
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率
        self.old_providers_count = old_providers_count  # 已建充电站数量
        self.low = low
        self.up = up
        self.THROD = THROD
        self.BEITA = BEITA
        self.PENTENTIAL_NP_C = PENTENTIAL_NP_C  # 搜索维度
        self.M = f_num  # 目标个数
        self.pop_size = pop_size  # 总群个体数量
        self.max_iteration = max_iter  # 迭代次数
        self.f_num=f_num
        self.is_builded_providers_adjust=is_builded_providers_adjust
        self.B_NP_CF=B_NP_CF
        self.NP_C=NP_C
        self.B_NP_SF=B_NP_SF
        self.NP_S=NP_S
        self.E_NEW_SUM=E_NEW_SUM
        self.SMALL_INF=SMALL_INF
        self.BIG_INF=BIG_INF


    # 主函数
    def excute_nsga3(self, demands_provider_np, demands_pdd_np, provider_id_list):
        # 首先获取原有历史构建的快速充电设施的数量
        self.E_OLD_SUM = np.sum(demands_provider_np[:, 0, 0])
        # 产生一致性的参考点
        Z, actural_popsize = self.uniformpoint(self.pop_size)  # 生成一致性的参考解
        # 随机初始化种群
        popu = self.initial_population(actural_popsize)
        popu_fitness = self.fitness(demands_provider_np, demands_pdd_np, popu)  # 计算适应度函数值
        # 绘制PF
        fig1 = plt.figure()
        plt.xlim([0, self.M])
        for i in range(popu_fitness.shape[0]):
            plt.plot(np.array(popu_fitness[i, :]))
        plt.show()
        Zmin = np.array(np.min(popu_fitness,0)).reshape(1, self.M)  # 求理想点
        # 迭代过程
        for i in range(self.max_iteration):
            print("第{name}次迭代".format(name=i))
            matingpool = random.sample(range(actural_popsize), actural_popsize)
            # off_spring_popu = self.crossover_and_mutation(popu[matingpool,:], self.t1, self.t2, self.pc, self.pm)# 遗传算子,模拟二进制交叉和多项式变异
            new_population_from_selection_mutation = self.cross_and_mutation(popu[matingpool,:])
            off_spring_fitness = self.fitness(demands_provider_np, demands_pdd_np, new_population_from_selection_mutation)  # 计算适应度函数
            double_popu = copy.deepcopy(np.vstack((popu, new_population_from_selection_mutation)))
            Zmin = np.array(np.min(np.vstack((Zmin, off_spring_fitness)), 0)).reshape(1, self.M)  # 更新理想点
            popu = self.envselect(demands_provider_np, demands_pdd_np, double_popu,actural_popsize, Z, Zmin)
            popu_fitness = self.fitness(demands_provider_np, demands_pdd_np,  popu)
        # 绘制PF
        fig1 = plt.figure()
        plt.xlim([0, self.M])
        for i in range(popu_fitness.shape[0]):
            plt.plot(np.array(popu_fitness[i, :]))
        plt.show()



def main_565_186_CUF_SUF_Jianye():
    # 参数设置  代
    N_GENERATIONS2 = 100  # 迭代次数
    # 区
    POP_SIZE2 = 400  # 种群大小
    pc2 = 0.8  # 交叉概率
    pm2 = 0.1  # 变异概率
    f_num = 5

    # # # # 测试4：已建服务设施不调整，非固定provider count，非固定 provider scale
    # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    B_NP_CF = False
    B_NP_C = None  # 无效的变量，不会被用到的
    # 每个provider的规模是否相同
    B_NP_SF = False
    B_NP_S = None   # 无效的变量，不会被用到的
    # 总规模
    E_seting = 200

    #### 测试建邺快速充电桩的数据
    DEMANDS_COUNT = 72  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 146
    OLD_PROVIDERS_COUNT = 40
    BEITA = 0.8
    # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 0.3  # 有效距离或者时间的阈值
    low2 = 2  # 最低阈值，至少建设1个
    up2 = 20  # 最高阈值，最多建设20个
    SMALL_INF=1e-6
    BIG_INF=1e6
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_jy_time1000"  # COLLECTION 名称

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_PS_JY_TIME(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C,SMALL_INF)  # 必须要先创建索引，才可以执行
    print("总人口",np.nansum(demands_pdd_np[:, 0]))
    print(provider_id_list)
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA3(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NP_C, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, B_NP_CF,
                      B_NP_C, B_NP_SF,
                      B_NP_S, E_seting,SMALL_INF,BIG_INF)
    nsga2_obj.excute_nsga3(demands_provider_np, demands_pdd_np, provider_id_list)


# NSGA2入口
if __name__ == '__main__':
    main_565_186_CUF_SUF_Jianye()

