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
sns.set(style='whitegrid', context='notebook')
# sns.set(style="ticks", color_codes=True)
import copy



class NSGA2():
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
        :param cipf_np:
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
            # y4_cover_people[i] = self.calcuate_global_cover_people(demands_provider_np_temp,solution_join)
            # 建设成本，越小越好
            y5_cost[i]=self.calculate_global_cost_numpy(solution_join)
        # 统一转成最小化问题
        # pop_fitness = np.vstack((1e-5*y1_driving_time,y2_waiting_time, 100 * y3_equlity, 2e7 / y4_cover_people,0.0001*y5_cost)).T
        pop_fitness = np.vstack((1e-5*y1_driving_time, 1e-2 * y3_inequlity, 1e-4*y5_cost)).T
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
            # 充电桩供给量 M_J
            M_J=demands_provider_np[:,0,0][mask]
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
        return 1000000/np.nansum(vj)

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
    def initial_population(self):
        """
        初始化种群：
        方式1：定个数is_new_building_providers_count_fixed,定规模 is_every_new_building_providers_scale_equal、
        :return:
        """
        population = np.full(shape=(self.pop_size, self.x_dim), fill_value=0,dtype=np.int)
        for m in range(self.pop_size):
            if self.B_NP_CF ==True:
                #count fixed
                if self.B_NP_SF==True:
                    # count fixed，scale fixed
                    population[m, :] = self.create_solution_CF_SF(self.x_dim, self.NP_C, self.NP_S)
                else:
                    # count fixed，scale unfixed
                    population[m, :] = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_NEW_SUM, k=self.NP_C)
            elif self.B_NP_CF ==False:
                #非固定的provider count
                if self.B_NP_SF:
                    #count unfixed ，scale fixed 其本质上就是CF_SF，因为我们的总数量是规定的
                    population[m, :] = self.create_solution_CF_SF(self.x_dim, int(self.E_NEW_SUM / self.NP_S), self.NP_S)
                else:
                    # count unfixed ，scale unfixed
                    #首先，随机明确新建 count，然后转换为count fixed，scale unfixed
                    np_count = np.random.randint(low=int(self.E_NEW_SUM / self.up), high=int(self.E_NEW_SUM / self.low), size=1)[0]
                    population[m, :]=self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_NEW_SUM, k=np_count)
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
        choic_index_list = np.random.choice(a=[i for i in range(x_dim)], size=k,
                                            replace=False)
        creating_mark = np.full(shape=(x_dim), fill_value=0)
        creating_mark[choic_index_list] = 1
        solution = np.random.randint(low=lower, high=up + 1, size=(x_dim))  # 新建的设施
        solution = creating_mark * solution
        #求平
        return self.balance_E_setting(E_setting, choic_index_list, lower, solution, up)


    def balance_E_setting(self, E_setting, choic_index_list, low, solution, up):
        """
        ????????????????????????????????
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
        off_spring = np.full(shape=(population.shape[0], self.x_dim), fill_value=0)
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
            solution = self.create_solution_CF_SF(self.x_dim, self.NP_C, self.NP_S)
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
            solution = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_NEW_SUM, k=self.NP_C)
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
            solution = self.create_solution_CF_SF(self.x_dim, int(self.E_NEW_SUM / self.NP_S), self.NP_S)
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
            solution = self.create_solution_CF_SUF(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_NEW_SUM, k=k)
        return solution

    # </editor-fold>
    ##############################################end：population#########################################################

    # <editor-fold desc="非支配排序、拥挤度算子、选择">
    def conbine_children_parent(self, population, new_population_from_selection_mutation):
        """
        # 父代种群和子代种群合并,pop*2
        :param population:
        :param new_population_from_selection_mutation:
        :return:
        """
        population_child_conbine = np.zeros((2 * self.pop_size, self.x_dim))  # self.population
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                population_child_conbine[i][j] = population[i][j]
                population_child_conbine[i + self.pop_size][j] = new_population_from_selection_mutation[i][j]
        return population_child_conbine

    def select_population_from_parent(self, population_child_conbine, fronts, crowding_distance):
        """
        # 根据排序和拥挤度计算，选取新的父代种群 pop*2 到 pop*1
        :param population_child_conbine:
        :param fronts:
        :param crowding_distance:
        :return:
        """
        population = np.zeros((self.pop_size, self.x_dim))  # 选取新的种群
        try:
            a = len(fronts[0])  # Pareto前沿面第一层 个体的个数
            if a >= self.pop_size:
                for i in range(self.pop_size):
                    population[i] = population_child_conbine[fronts[0][i]]
            else:
                d = []  # 用于存放前b层个体
                i = 1
                while a < self.pop_size:
                    c = a  # 新种群内 已经存放的个体数目    *列
                    if i >= len(fronts):
                        break
                    a += len(fronts[i])
                    for j in range(len(fronts[i - 1])):
                        d.append(fronts[i - 1][j])
                    b = i  # 第b层不能放，超过种群数目了    *行
                    i = i + 1
                # 把前c个放进去
                for j in range(c):
                    population[j] = population_child_conbine[d[j]]
                temp = np.zeros((len(fronts[b]), 2))  # 存放拥挤度和个体序号
                for i in range(len(fronts[b])):
                    temp[i][0] = crowding_distance[fronts[b][i]]
                    temp[i][1] = fronts[b][i]
                temp = sorted(temp.tolist())  # 拥挤距离由小到大排序
                for i in range(self.pop_size - c):
                    population[c + i] = population_child_conbine[int(temp[len(temp) - i - 1][1])]
                    # 按拥挤距离由大到小填充直到种群数量达到 pop
        except Exception as E_results:
            print("捕捉有异常：", E_results)
            population = population_child_conbine[0:self.pop_size, :]
        return population

    def non_donminate(self, objectives_fitness):
        """
        非支配排序
        :param population:
        :param objectives_fitness:
        :return:
        """
        fronts = []  # Pareto前沿面
        fronts.append([])
        set_sp = []
        npp = np.zeros(objectives_fitness.shape[0])
        rank = np.zeros(objectives_fitness.shape[0])
        for i in range(objectives_fitness.shape[0]):
            temp = []
            for j in range(objectives_fitness.shape[0]):
                if j != i:
                    # # temp=[]
                    # if j != i:
                    # 将传统的逐一判断方式摈弃掉
                    #     if (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1] and objectives_fitness[j][2] > objectives_fitness[i][2]) or (
                    #         objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1] and objectives_fitness[j][2] > objectives_fitness[i][2]) or (
                    #         objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1] and objectives_fitness[j][2] >= objectives_fitness[i][2]):
                    #         temp.append(j)
                    #     elif (objectives_fitness[i][0] >= objectives_fitness[j][0] and objectives_fitness[i][1] > objectives_fitness[j][1] and objectives_fitness[i][2] > objectives_fitness[j][2]) or (
                    #             objectives_fitness[i][0] > objectives_fitness[j][0] and objectives_fitness[i][1] >= objectives_fitness[j][1] and objectives_fitness[i][2] > objectives_fitness[j][2]) or (
                    #             objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1] and objectives_fitness[i][2] >= objectives_fitness[j][2]):
                    #         npp[i] += 1  # j支配 i，np+1
                    less = 0  # y'的目标函数值小于个体的目标函数值数目
                    equal = 0  # y'的目标函数值等于个体的目标函数值数目
                    greater = 0  # y'的目标函数值大于个体的目标函数值数目
                    for k in range(self.f_num):
                        if (objectives_fitness[i][k] < objectives_fitness[j][k]):
                            less = less + 1
                        elif (objectives_fitness[i][k] == objectives_fitness[j][k]):
                            equal = equal + 1
                        else:
                            greater = greater + 1
                    if (less == 0 and equal != self.f_num):
                        npp[i] += 1  # j支配 i，np+1
                    elif (greater == 0 and equal != self.f_num):
                        temp.append(j)
            set_sp.append(temp)  # i支配 j，将 j 加入 i 的支配解集里
            if npp[i] == 0:
                fronts[0].append(i)  # 个体序号
                rank[i] = 1  # Pareto前沿面 第一层级
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for j in range(len(fronts[i])):
                a = 0
                while a < len(set_sp[fronts[i][j]]):
                    npp[set_sp[fronts[i][j]][a]] -= 1
                    if npp[set_sp[fronts[i][j]][a]] == 0:
                        rank[set_sp[fronts[i][j]][a]] = i + 2  # 第二层级
                        temp.append(set_sp[fronts[i][j]][a])
                    a = a + 1
            i = i + 1
            fronts.append(temp)
        return fronts

    def crowd_distance(self, fronts, objectives_fitness):
        """
         # 拥挤度计算，前沿面每个个体的拥挤度
        :param fronts:
        :param objectives_fitness:
        :return:
        """
        crowding_distance = np.zeros(2 * self.pop_size)
        for i in range(len(fronts) - 1):  # fronts最后一行为空集
            temp1 = np.zeros((len(fronts[i]), 2))
            temp2 = np.zeros((len(fronts[i]), 2))
            temp3 = np.zeros((len(fronts[i]), 2))
            for j in range(len(fronts[i])):
                temp1[j][0] = objectives_fitness[fronts[i][j]][0]  # f1赋值
                temp1[j][1] = fronts[i][j]
                temp2[j][0] = objectives_fitness[fronts[i][j]][1]  # f2赋值
                temp2[j][1] = fronts[i][j]
                temp3[j][0] = objectives_fitness[fronts[i][j]][2]  # f3赋值
                temp3[j][1] = fronts[i][j]

            # temp3 = temp1.tolist()
            # temp4 = temp2.tolist()
            temp1 = sorted(temp1.tolist())  # f1排序
            temp2 = sorted(temp2.tolist())  # f2排序
            temp3 = sorted(temp3.tolist())  # f3排序
            crowding_distance[int(temp1[0][1])] = float('inf')
            crowding_distance[int(temp1[len(fronts[i]) - 1][1])] = float('inf')
            f1_min = temp1[0][0]
            f1_max = temp1[len(fronts[i]) - 1][0]
            f2_min = temp2[0][0]
            f2_max = temp2[len(fronts[i]) - 1][0]
            f3_min = temp3[0][0]
            f3_max = temp3[len(fronts[i]) - 1][0]
            a = 1
            while a < len(fronts[i]) - 1:
                # 个体i的拥挤度等于 f1 + f2 + f3
                crowding_distance[int(temp1[a][1])] = (temp1[a + 1][0] - temp1[a - 1][0]) / (
                            f1_max - f1_min + 0.0000001) + \
                                                      (temp2[a + 1][0] - temp2[a - 1][0]) / (
                                                                  f2_max - f2_min + 0.0000001) + \
                                                      (temp3[a + 1][0] - temp3[a - 1][0]) / (
                                                                  f3_max - f3_min + 0.0000001)
                a += 1
        return crowding_distance
    # </editor-fold>

    # <editor-fold desc="可视化与测试">
    def show_initial_population_fitness_graph(self, demands_provider_np, demands_pdd_np, initial_population):  # 画图
        """
        绘制结果图
        :param demands_provider_np:
        :param demands_pdd_np:
        :param initial_population:
        :return:
        """
        # 评价函数
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, initial_population)
        if self.f_num==2:
            x = []
            y = []
            for i in range(objectives_fitness.shape[0]):
                x.append(objectives_fitness[i][0])
                y.append(objectives_fitness[i][1])
            ax = plt.subplot(111)
            plt.scatter(x, y)
            # plt.plot(,'--',label='')
            plt.axis([3.5, 6, 0, 5])
            xmajorLocator = MultipleLocator(0.1)
            ymajorLocator = MultipleLocator(0.1)
            ax.xaxis.set_major_locator(xmajorLocator)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.xlabel('1000 * inequality')
            plt.ylabel('20000000 / cover_population')
            plt.title('Initial Population Fitness')
            plt.grid()
            plt.show()
            # plt.savefig('nsga2 ZDT2 Pareto Front 2.png')
        elif self.f_num==3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("travel time")
            ax.set_ylabel("inequality")
            ax.set_zlabel("cost")
            ax.set_title('Initial Population Fitness')
            type1 = ax.scatter(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2], c='g')
            # plt.legend((type1), (u'Non-dominated solution'))
            plt.show()
            # 绘制Pair graph 散点图
            self.show_pair_plot_of_fitness(objectives_fitness)

    def show_pair_plot_of_fitness(self, objectives_fitness):
        df4 = pd.DataFrame(objectives_fitness)
        df4.loc[:, "initial"] = "initial"
        df4.columns = ['1000 * inequality', '20000000/cover_population', '0.0001 * cost', "initial"]
        sns.pairplot(df4, hue='initial', height=2.5, markers=["o"]);
        plt.show()

    def show_fitness_graph_and_pairgraph(self, demands_provider_np, demands_pdd_np, population_front0):  # 画图
        """
        绘制结果图
        :param demands_provider_np:
        :param demands_pdd_np:
        :param population:
        :return:
        """
        # 评价函数
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population_front0)
        if self.f_num==2:
            F1 = []  # equlity
            F2 = []  # cover people
            for i in range(objectives_fitness.shape[0]):
                F1.append(objectives_fitness[i][0])
                F2.append(objectives_fitness[i][1])
            ax = plt.subplot(111)
            plt.scatter(F1, F2)
            # plt.plot(,'--',label='')
            plt.axis([np.min(objectives_fitness[:, 0]), np.max(objectives_fitness[:, 0]), np.min(objectives_fitness[:, 1]),np.max(objectives_fitness[:, 1])])
            # xmajorLocator = MultipleLocator(0.1)
            # ymajorLocator = MultipleLocator(0.1)
            # ax.xaxis.set_major_locator(xmajorLocator)
            # ax.yaxis.set_major_locator(ymajorLocator)
            plt.xlabel('1000 * equlity')
            plt.ylabel('20000000 / cover_population')
            plt.title('理论图 多目标求解')
            plt.grid()
            plt.show()
            # plt.savefig('nsga2 ZDT2 Pareto Front 2.png')
        elif self.f_num==3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("1000 * inequality")
            ax.set_ylabel("20000000 / cover_population")
            ax.set_zlabel("0.0001 * cost")
            ax.set_title('Pareto Population Fitness')
            type1 = ax.scatter(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2], c='g')
            # plt.legend((type1), (u'Non-dominated solution'))
            plt.show()
            self.show_pair_plot_of_fitness(objectives_fitness)
            # 将结果存储下来
            np.savez("./log/jianye_solutions", population_front0=population_front0,objectives_fitness=objectives_fitness)


    def show_frequency_bar_graph(self, result_popu, provider_id_list):
        """
        输出Pareto非劣解的结果，并采用频率分布图表示
        :param result_popu:
        :return:
        """
        # 输出所有的Provider id list
        print(provider_id_list[2:])  # 前两个11,13是已经建设过得了，所以从中去掉
        popu_index = []
        # 计算得新建的provider ID list
        created_provider_id_list_from_result_popu = []
        for i in range(result_popu.shape[0]):
            popu_index.append(np.argwhere(result_popu[i, :] >0).flatten())
            result_provider_id_solution = []
            for j in range(len(popu_index[i])):
                result_provider_id_solution.append(provider_id_list[2:][popu_index[i][j]])
            created_provider_id_list_from_result_popu.append(result_provider_id_solution)

        # TESTCODE 测试代码 将结果序列输出
        created_provider_id_list_from_result_popu = np.array(created_provider_id_list_from_result_popu)
        # for i in range(created_provider_id_list_from_result_popu.shape[0]):
        #     print(created_provider_id_list_from_result_popu[i, :])

        # 将序列做从小到大的排序，以便于后面求unique
        created_provider_id_list_from_result_popu = np.sort(created_provider_id_list_from_result_popu, axis=1)
        # unique并返回统计的数量
        uniques = np.unique(created_provider_id_list_from_result_popu, return_counts=True, axis=0)
        # 进行制图表达
        x_label = []
        x_label = [str(rowitem) for rowitem in uniques[0]]
        plt.bar(x_label, uniques[1])
        plt.xticks(size=14, rotation=60)
        plt.yticks(size=14)
        plt.show()
        # 将frequency_result存储到文件夹中
        np.savez("./log/jianye_frequency", frequency=uniques[0],
                 frequency_count=uniques[1])

    # def evalution_result(self):
    #     # --------------------Coverage(C-metric)---------------------
    #     A = PP #
    #     B = chromo
    #     number = 0
    #     for i in range(len(B)):
    #         nn = 0
    #         for j in range(len(A)):
    #             if (Dominate(A[j], B[i])):
    #                 nn = nn + 1  # B[i]被A支配的个体数目+1
    #         if (nn != 0):
    #             number = number + 1
    #     C_AB = float(number / len(B))
    #     print("C_AB：%2f" % C_AB)
    #     # -----Distance from Representatives in the PF(D-metric)-----
    #     A = chromo
    #     P = PP
    #     min_d = 0
    #     for i in range(len(P)):
    #         temp = []
    #         for j in range(len(A)):
    #             dd = 0
    #             for k in range(f_num):
    #                 dd = dd + float((P[i][k] - A[j].f[k]) ** 2)
    #             temp.append(math.sqrt(dd))
    #         min_d = min_d + np.min(temp)
    #     D_AP = float(min_d / len(P))
    #     print("D_AP：%2f" % D_AP)

    # </editor-fold>

    def __init__(self, pc, pm, low, up, old_providers_count, THROD, BEITA, x_dim, pop_size, max_iter, f_num,is_builded_providers_adjust, B_NP_CF, NP_C,B_NP_SF,NP_S, E_NEW_SUM,SMALL_INF,BIG_INF):
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率
        self.old_providers_count = old_providers_count  # 已建充电站数量
        self.low = low
        self.up = up
        self.THROD = THROD
        self.BEITA = BEITA
        self.x_dim = x_dim  # 搜索维度
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

    def excute_nsga2(self, demands_provider_np, demands_pdd_np, provider_id_list):
        """
        # 主程序
        :param demands_provider_np:
        :param demands_pdd_np:
        :return:
        """
        # 首先获取原有历史构建的快速充电设施的数量
        self.E_OLD_SUM=np.sum(demands_provider_np[:,0,0])
        # 初始化种群
        population=self.initial_population()
        # 绘制非劣解的fitness关联图
        self.show_initial_population_fitness_graph(demands_provider_np, demands_pdd_np, population)
        for i in range(self.max_iteration):
            matingpool = random.sample(range(self.pop_size), self.pop_size)
            new_population_from_selection_mutation = self.cross_and_mutation(population[matingpool,:])
            # 父代与子代种群合并
            population_child_conbine = self.conbine_children_parent(population,new_population_from_selection_mutation)
            # 评价函数
            objectives_fitness = self.fitness(demands_provider_np,demands_pdd_np,population_child_conbine)
            # 快速非支配排序
            fronts=self.non_donminate(objectives_fitness)
            # fronts=self.fast_non_dominated_sort(objectives_fitness)
            # 拥挤度计算
            crowding_distance=self.crowd_distance(fronts,objectives_fitness)
            # 根据Pareto等级和拥挤度选取新的父代种群，选择交叉变异
            population=self.select_population_from_parent(population_child_conbine, fronts, crowding_distance)
            print(i,"代")
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population)
        fronts = self.non_donminate(objectives_fitness)
        # fronts=self.fast_non_dominated_sort(objectives_fitness)
        population_front0 = population[fronts[0]]
        # 绘制非劣解的fitness关联图，并将结果存储下来
        self.show_fitness_graph_and_pairgraph(demands_provider_np, demands_pdd_np, population_front0)
        # if scale fixed，查看评率分布才是有意义的。
        if self.B_NP_SF==True:
            # 绘制频率分布图并将frequency_result存储到文件夹中
            self.show_frequency_bar_graph(population_front0, provider_id_list)


def main_55_CF_SF():
    # 参数设置  代
    N_GENERATIONS2 = 500  # 迭代次数
    # 区
    POP_SIZE2 = 200  # 种群大小
    pc2 = 0.6  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 4
    DEMANDS_COUNT = 9  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 14
    OLD_PROVIDERS_COUNT = 2
    BEITA = 1
    THROD = 3000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory"  # COLLECTION 名称
    # # #测试1：已建服务设施不调整，固定provider count，固定 provider scale
    # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    B_NP_CF = True
    B_NP_C = 2
    # 每个provider的规模是否相同
    B_NP_SF = True
    B_NP_S = 10
    # 总规模
    E_seting = 20

    # ### 测试3：已建服务设施不调整，非固定provider count，固定 provider scale
    # ###已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # B_NP_CF = False
    # B_NP_C = None  #无效的变量，不会被用到的
    # # 每个provider的规模是否相同
    # B_NP_SF = True
    # B_NP_S = 10
    # # 总规模
    # E_seting = 20   #要求E_seting与every_new_building_providers_scale之间呈整数倍关系；

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_np_theory(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C)  # 必须要先创建索引，才可以执行
    print(provider_id_list)
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NP_C, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, B_NP_CF,
                      B_NP_C, B_NP_SF,
                      B_NP_S, E_seting)
    nsga2_obj.excute_nsga2(demands_provider_np, demands_pdd_np, provider_id_list)

def main_55_CF_SUF():
    # 参数设置  代
    N_GENERATIONS2 = 500  # 迭代次数
    # 区
    POP_SIZE2 = 400  # 种群大小
    pc2 = 0.6  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 3
    DEMANDS_COUNT = 9  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 14
    OLD_PROVIDERS_COUNT = 2
    BEITA = 1
    THROD = 3000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory"  # COLLECTION 名称

    # # # 测试2：已建服务设施不调整，固定provider count，不固定 provider scale
    # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    B_NP_CF = True
    B_NP_C = 2
    # 每个provider的规模是否相同
    B_NP_SF = False
    B_NP_S = None   #无效的变量，不会被用到的
    # 总规模
    E_seting = 20

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_np_theory(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C)  # 必须要先创建索引，才可以执行
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NP_C, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, B_NP_CF,
                      B_NP_C, B_NP_SF,
                      B_NP_S, E_seting)
    nsga2_obj.excute_nsga2(demands_provider_np, demands_pdd_np, provider_id_list)

def main_55_CUF_SUF():
    # 参数设置  代
    N_GENERATIONS2 = 500  # 迭代次数
    # 区
    POP_SIZE2 = 400  # 种群大小
    pc2 = 0.6  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 3
    DEMANDS_COUNT = 9  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 14
    OLD_PROVIDERS_COUNT = 2
    BEITA = 1
    THROD = 3000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory"  # COLLECTION 名称

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
    E_seting = 100

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_np_theory(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C)  # 必须要先创建索引，才可以执行
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NP_C, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, B_NP_CF,
                      B_NP_C, B_NP_SF,
                      B_NP_S, E_seting)
    nsga2_obj.excute_nsga2(demands_provider_np, demands_pdd_np, provider_id_list)

def main_565_186_CUF_SUF_Jianye():
    # 参数设置  代
    N_GENERATIONS2 = 100  # 迭代次数
    # 区
    POP_SIZE2 = 400  # 种群大小
    pc2 = 0.8  # 交叉概率
    pm2 = 0.1  # 变异概率
    f_num = 3

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
    up2 = 10  # 最高阈值，最多建设10个
    SMALL_INF=1e-6
    BIG_INF=1e8
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_jy_time1000"  # COLLECTION 名称

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_PS_JY_TIME(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C,SMALL_INF)  # 必须要先创建索引，才可以执行
    print(provider_id_list)
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NP_C, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, B_NP_CF,
                      B_NP_C, B_NP_SF,
                      B_NP_S, E_seting,SMALL_INF,BIG_INF)
    nsga2_obj.excute_nsga2(demands_provider_np, demands_pdd_np, provider_id_list)


# NSGA2入口
if __name__ == '__main__':
    main_565_186_CUF_SUF_Jianye()

