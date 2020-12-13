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
import a_mongo_operater_theory
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid', context='notebook')
# sns.set(style="ticks", color_codes=True)


class NSGA2():
    ##############################################begin:fitness#########################################################
    # <editor-fold desc="fitness">
    def fitness(self, demands_provider_np, demands_pdd_np, popu):
        """
        # fitness计算，全局的适应度函数
        :param demands_np:
        :param cipf_np:
        :param popu:
        :return:
        """
        actural_popsize = popu.shape[0]
        y1_values_double = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 与居民区可达性
        y2_values_double = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 与居民区公平性
        y5_values_double = np.full(shape=(actural_popsize,), fill_value=1, dtype=np.float32)  # 覆盖人口
        for i in range(actural_popsize):
            solution = popu[i, :]
            solution_join = np.hstack((np.full(shape=self.old_providers_count, fill_value=0.01), solution))  # 把前面的已建部分的0.01补齐；
            # demands_provider_np 计算vj
            self.update_provider_vj(demands_provider_np, demands_pdd_np, solution_join)
            self.calculate_single_provider_gravity_value_np(demands_provider_np, demands_pdd_np, solution_join)
            # 居民区 可达性适应度值，该值越大越好
            y1_values_double[i] = self.calculate_global_accessibility_numpy(demands_pdd_np)
            # 居民区  计算公平性数值，该值越小表示越公平
            y2_values_double[i] = self.calculate_global_equality_numpy(demands_pdd_np)
            # 覆盖人口，越大越好
            y5_values_double[i] = self.calcuate_global_cover_people(demands_provider_np)
        # 统一转成最小化问题
        pop_fitness = np.vstack((1 / y1_values_double, 1000 * y2_values_double, 20 / y5_values_double)).T
        return pop_fitness

    def calculate_global_accessibility_numpy(self, demands_pdd_np):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        return np.nansum(demands_pdd_np[:, 1])

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

    def calcuate_global_cover_people(self, demands_provider_np):
        """
        获取覆盖全局人口的总和
        :param demands_np:
        :return:
        """
        vj = demands_provider_np[:, 0, 2]
        vj[np.isinf(vj)] = 0
        vj[np.isnan(vj)] = 0
        return np.nansum(vj)  # 获取到每一个的provider的vj值，然后求和，之所以用nan，是为了过滤掉nan的值

    def create_providers_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        构建以providers为主导的np数组
        :param demands_np:
        :param new_providers_count:
        :param solution_join:
        :return:
        """
        providers_np = np.full((demands_provider_np.shape[0], demands_provider_np.shape[1], 2), 0.000000001)
        providers_np[:, :, 0] = np.tile(demands_pdd_np[:, 0], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 1] = demands_provider_np[:, :, 1]  # D_T
        # 将solution部分增加进去
        mask = solution_join == 0
        mask = mask.reshape(mask.shape[0], 1)
        mask2 = np.tile(mask, (1, demands_provider_np.shape[1]))
        providers_np[:, :, 0][mask2] = 0
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

    # </editor-fold>
    ##############################################end:fitness#########################################################

    ##############################################begin：population#########################################################
    # <editor-fold desc="population crossover mutation">
    def initial_population(self):
        """
           初始化种群： # 方式1：定规模、定个数
        :return:
        """
        population = np.full(shape=(self.pop_size, self.x_dim), fill_value=0,dtype=np.int)
        for m in range(self.pop_size):
            if self.is_new_building_providers_count_fixed:
                #固定的provider count
                if self.is_every_new_building_providers_scale_equal:
                    # 固定的provider count，固定的provider 规模
                    population[m, :] = self.create_solution_fixcount_fixscale(self.x_dim, self.new_building_providers_count_fixed, self.every_new_building_providers_scale)
                else:
                    # 固定的provider count，非固定的provider规模
                    population[m, :] = self.create_solution_fixcount_unfixscale(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_setting, k=self.new_building_providers_count_fixed)
            else:
                #非固定的provider count
                if self.is_every_new_building_providers_scale_equal:
                    # 非固定的provider count，固定的provider 规模
                    population[m, :] = self.create_solution_fixcount_fixscale(self.x_dim, int(self.E_setting / self.every_new_building_providers_scale), self.every_new_building_providers_scale)
                else:
                    # 非固定的provider count，非固定的provider规模
                    #首先明确新建多少个provider count
                    k = np.random.randint(low=int(self.E_setting/self.up) , high=int(self.E_setting/self.low) , size=1)[0]  # 新建的设施
                    population[m, :]=self.create_solution_fixcount_unfixscale(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_setting, k=k)
            # #下面为测试代码，测试生成的个体中选择新建provider的个数是否与self.new_building_providers_count_fixed一致
            # if np.argwhere(population[m, :] == 10).flatten().shape[0] != self.new_building_providers_count_fixed:
            #     print("不等于self.new_building_providers_count_fixed")
        return population

    def create_solution_fixcount_fixscale(self, x_dim, k, scale):
        """
        固定provider count，固定provider scale
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

    def create_solution_fixcount_unfixscale(self, lower, up, x_dim, E_setting, k):
        """
        固定provider count，非固定provider scale
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

    def balance_E_setting(self, E_setting, choic_index_list, lower, solution, up):
        """
        将不平整的部分平整输出
        :param E_setting:
        :param choic_index_list:
        :param lower:
        :param solution:
        :param up:
        :return:
        """
        solution = (solution * (E_setting / np.sum(solution))).astype(np.int)  # 规模介于up和low之间
        # 会存在部分因最大值整数约束，而超限的数值，需要先处理到【low,up】范围内
        while np.sum(solution[np.where((solution < lower) & (solution != 0))]) > 0 or np.sum(
                solution[solution > up]) > 0:
            solution[np.where((solution < lower) & (solution != 0))] += 1
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
                if solution[adjust_index] > lower and solution[adjust_index] < up:
                    solution[adjust_index] += adjust_mark
                if delate > 0 and solution[adjust_index] == up:
                    solution[adjust_index] += adjust_mark
                if delate < 0 and solution[adjust_index] == lower:
                    solution[adjust_index] += adjust_mark
            # if iteation_count<100:
            #     iteation_count+=1
            # else:
            #     break
        return solution

    def cross_and_mutation(self, population):
        # 选择、交叉、变异，生成子代种群
        off_spring = np.full(shape=(population.shape[0], self.x_dim), fill_value=0)
        for i in range(population.shape[0]):
            a1 = random.randint(0, population.shape[0] - 1)
            b1 = random.randint(0, population.shape[0] - 1)
            if self.is_new_building_providers_count_fixed:
                # 固定的provider count
                if self.is_every_new_building_providers_scale_equal:
                    # 固定的provider count，固定的provider 规模
                    solution = self.cross_fixcount_fixscale(population[a1, :], population[b1, :], self.pc)
                    off_spring[i, :] = self.mutation_fixcount_fixscale(solution, self.pm)
                else:
                    # 固定的provider count，非固定的provider规模
                    solution = self.cross_fixcount_unfixscale(a1, b1, population)
                    off_spring[i, :] = self.mutation_fixcount_unfixscale(solution, self.pm)
            else:
                # 非固定的provider count
                if self.is_every_new_building_providers_scale_equal:
                    # 非固定的provider count，固定的provider 规模
                    solution = self.cross_unfixcount_fixscale(population[a1, :], population[b1, :], self.pc)
                    off_spring[i, :] = self.mutation_unfixcount_fixscale(solution, self.pm)
                else:
                    # 非固定的provider count，非固定的provider规模
                    solution = self.cross_unfixcount_unfixscale(population[a1, :], population[b1, :], self.pc)
                    off_spring[i, :] = self.mutation_unfixcount_unfixscale(solution, self.pm)
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

    def cross_fixcount_fixscale(self, solution_a, solution_b, pc):
        """
        交叉算子
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        crossover_prob = random.random()
        if crossover_prob < pc:
            #取出有值的位置index list
            count=int(self.new_building_providers_count_fixed/2)
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

    def cross_fixcount_unfixscale(self, a1, b1, population):
        solution = self.cross_fixcount_fixscale(population[a1, :], population[b1, :], self.pc)
        # 求平
        if np.sum(solution) != self.E_setting:
            choic_index_list = np.argwhere(solution > 0).flatten()
            solution = self.balance_E_setting(self.E_setting, choic_index_list, self.low, solution, self.up)
        return solution

    def cross_unfixcount_fixscale(self, solution_a, solution_b, pc):
        """
        交叉算子
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        crossover_prob = random.random()
        if crossover_prob < pc:
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

    def cross_unfixcount_unfixscale(self, solution_a, solution_b, pc):
        """
        交叉算子
        :param solution_a:
        :param solution_b:
        :param pc:
        :return:
        """
        solution = self.cross_unfixcount_fixscale(solution_a, solution_b, self.pc)
        # 求平
        if np.sum(solution) != self.E_setting:
            choic_index_list = np.argwhere(solution > 0).flatten()
            solution = self.balance_E_setting(self.E_setting, choic_index_list, self.low, solution, self.up)
        return  solution

    def mutation_fixcount_fixscale(self, solution, pm):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < pm:
            solution = self.create_solution_fixcount_fixscale(self.x_dim, self.new_building_providers_count_fixed, self.every_new_building_providers_scale)
        return solution

    def mutation_fixcount_unfixscale(self, solution, pm):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < pm:
            # 固定的provider count，非固定的provider规模
            solution = self.create_solution_fixcount_unfixscale(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_setting, k=self.new_building_providers_count_fixed)
        return solution

    def mutation_unfixcount_fixscale(self, solution, pm):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < pm:
            # 非固定的provider count，固定的provider 规模
            solution = self.create_solution_fixcount_fixscale(self.x_dim, int(self.E_setting / self.every_new_building_providers_scale), self.every_new_building_providers_scale)
        return solution

    def mutation_unfixcount_unfixscale(self, solution, pm):
        """
        变异算子
        :param solution:
        :param pm:
        :return:
        """
        mutation_prob = random.random()
        if mutation_prob < pm:
            # 非固定的provider count，非固定的provider规模
            # 首先明确新建多少个provider count
            k = np.random.randint(low=int(self.E_setting / self.up), high=int(self.E_setting / self.low), size=1)[0]  # 新建的设施
            solution = self.create_solution_fixcount_unfixscale(lower=self.low, up=self.up, x_dim=self.x_dim, E_setting=self.E_setting, k=k)
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

    def non_donminate(self, population, objectives_fitness):
        """
        非支配排序
        :param population:
        :param objectives_fitness:
        :return:
        """
        fronts = []  # Pareto前沿面
        fronts.append([])
        set_sp = []
        npp = np.zeros(population.shape[0])
        rank = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            temp = []
            for j in range(population.shape[0]):
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
    def draw_initial_population_fitness_graph(self, demands_provider_np, demands_pdd_np, initial_population):  # 画图
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
            plt.xlabel('inequality')
            plt.ylabel('cover population')
            plt.title('Initial Population Fitness')
            plt.grid()
            plt.show()
            # plt.savefig('nsga2 ZDT2 Pareto Front 2.png')
        elif self.f_num==3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("accessibility")
            ax.set_ylabel("inequality")
            ax.set_zlabel("cover population")
            ax.set_title('Initial Population Fitness')
            type1 = ax.scatter(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2], c='g')
            # plt.legend((type1), (u'Non-dominated solution'))
            plt.show()

            self.pair_plot_for_fitness(objectives_fitness)

    def pair_plot_for_fitness(self, objectives_fitness):
        df4 = pd.DataFrame(objectives_fitness)
        df4.loc[:, "initial"] = "initial"
        df4.columns = ['accessibility', 'inequality', 'cover_population', "initial"]
        sns.pairplot(df4, hue='initial', height=2.5, markers=["o"]);
        plt.show()

    def draw_fitness_graph(self, demands_provider_np, demands_pdd_np, population):  # 画图
        """
        绘制结果图
        :param demands_provider_np:
        :param demands_pdd_np:
        :param population:
        :return:
        """
        # 评价函数
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population)
        fronts=self.non_donminate(population,objectives_fitness)
        # fronts=self.fast_non_dominated_sort(objectives_fitness)
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population[fronts[0]])

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
            plt.xlabel('inequality')
            plt.ylabel('cover population')
            plt.title('Pareto Population Fitness')
            plt.grid()
            plt.show()
            # plt.savefig('nsga2 ZDT2 Pareto Front 2.png')
        elif self.f_num==3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("accessibility")
            ax.set_ylabel("inequality")
            ax.set_zlabel("cover population")
            ax.set_title('Pareto Population Fitness')
            type1 = ax.scatter(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2], c='g')
            # plt.legend((type1), (u'Non-dominated solution'))
            plt.show()

            self.pair_plot_for_fitness(objectives_fitness)
        #将值输出出来
        for iii in range(len(population[fronts[0]])):
            print(population[fronts[0]][iii,:])
        objectives_fitness=objectives_fitness[objectives_fitness[:,0].argsort()]
        for jjj in range(objectives_fitness.shape[0]):
            print(objectives_fitness[jjj])
        return population[fronts[0]]

    def show_population_frequency_bar_graph(self,result_popu):
        """
        输出Pareto非劣解的结果，并采用频率分布图表示
        :param result_popu:
        :return:
        """
        # 输出popu结果
        print(provider_id_list[2:])
        popu_index = []
        # 输出provider ID
        result_provider_id_popu = []
        for i in range(result_popu.shape[0]):
            popu_index.append(np.argwhere(result_popu[i, :] == 10).flatten())
            result_provider_id_solution = []
            for j in range(len(popu_index[i])):
                result_provider_id_solution.append(provider_id_list[2:][popu_index[i][j]])
            result_provider_id_popu.append(result_provider_id_solution)
        # result_provider_id_popu=np.array(result_provider_id_popu)
        # for i in range(result_provider_id_popu.shape[0]):
        #     print(result_provider_id_popu[i,:])
        # 输出频率分布图
        frequency_result = self.test_get_count(result_provider_id_popu)   # 最后一列记录的是频率值
        self.test_show_bar(frequency_result)
        return frequency_result

    def test_get_count(self, result_provider_id_popu):
        '''
        计算每组解出现的次数
        :param result_provider_id_popu: 解集
        :return:
        '''
        for file in result_provider_id_popu:
            file.sort()
        count_list = []
        for file in result_provider_id_popu:
            count = []
            count.append(result_provider_id_popu.count(file))
            count_list.append(count)
        result = np.hstack((np.array(result_provider_id_popu), np.array(count_list)))
        uniques = np.unique(result, axis=0)  # 去重
        return uniques

    def test_show_bar(self,uniques):
        # 设置matplotlib正常显示中文和负号
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        # x_label = ['[' + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + ',' + str(i[3])+ ']' for i in uniques.tolist()]
        x_label = []
        for item in uniques.tolist():
            temp=""
            for jj in range(len(item)-1):
                if temp == "":
                    temp+=(str(item[jj]))
                else:
                    temp+=(","+str(item[jj]))
            x_label.append(temp)
        plt.bar(x_label, uniques[:, -1])
        plt.xticks(size=14, rotation=60)
        plt.yticks(size=14)
        plt.show()


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

    def __init__(self, pc, pm, low, up, old_providers_count,  THROD, BEITA, x_dim, pop_size, max_iter,f_num,
                 is_builded_providers_adjust, is_new_building_providers_count_fixed, new_building_providers_count_fixed,
                 is_every_new_building_providers_scale_equal,
                 every_new_building_providers_scale,E_setting ):
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
        self.is_new_building_providers_count_fixed=is_new_building_providers_count_fixed
        self.new_building_providers_count_fixed=new_building_providers_count_fixed
        self.is_every_new_building_providers_scale_equal=is_every_new_building_providers_scale_equal
        self.every_new_building_providers_scale=every_new_building_providers_scale
        self.E_setting=E_setting

    def excute(self, demands_provider_np, demands_pdd_np):
        """
        # 主程序
        :param demands_provider_np:
        :param demands_pdd_np:
        :return:
        """
        # 初始化种群
        population=self.initial_population()
        # 绘制非劣解的fitness关联图
        self.draw_initial_population_fitness_graph(demands_provider_np, demands_pdd_np, population)
        for i in range(self.max_iteration):
            matingpool = random.sample(range(self.pop_size), self.pop_size)
            new_population_from_selection_mutation = self.cross_and_mutation(population[matingpool,:])
            # 父代与子代种群合并
            population_child_conbine = self.conbine_children_parent(population,new_population_from_selection_mutation)
            # 评价函数
            objectives_fitness = self.fitness(demands_provider_np,demands_pdd_np,population_child_conbine)
            # 快速非支配排序
            fronts=self.non_donminate(population_child_conbine,objectives_fitness)
            # fronts=self.fast_non_dominated_sort(objectives_fitness)
            # 拥挤度计算
            crowding_distance=self.crowd_distance(fronts,objectives_fitness)
            # 根据Pareto等级和拥挤度选取新的父代种群，选择交叉变异
            population=self.select_population_from_parent(population_child_conbine, fronts, crowding_distance)
            print(i,"代")
        # 绘制非劣解的fitness关联图
        population_front0=self.draw_fitness_graph(demands_provider_np, demands_pdd_np, population)
        # 绘制频率分布图
        frequency_result=self.show_population_frequency_bar_graph(population_front0)
        # 将frequency_result存储到文件夹中
        np.savetxt("./frequency_result.txt", frequency_result, fmt='%f', delimiter=',')



def main_1():
    global provider_id_list
    # 参数设置  代
    N_GENERATIONS2 = 10  # 迭代次数
    # 区
    POP_SIZE2 = 200  # 种群大小
    pc2 = 0.25  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 3
    DEMANDS_COUNT = 9  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 14
    OLD_PROVIDERS_COUNT = 2
    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 10000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory"  # COLLECTION 名称


    # #测试1：已建服务设施不调整，固定provider count，固定 provider scale
    # # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = True
    new_building_providers_count_fixed = 2
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = True
    every_new_building_providers_scale = 10
    # 总规模
    E_seting = 20

    # # 测试2：已建服务设施不调整，固定provider count，不固定 provider scale
    # # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = True
    # new_building_providers_count_fixed = 2
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = False
    # every_new_building_providers_scale = None   #无效的变量，不会被用到的
    # # 总规模
    # E_seting = 20
    #
    # 测试3：已建服务设施不调整，非固定provider count，固定 provider scale
    # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = False
    # new_building_providers_count_fixed = None  #无效的变量，不会被用到的
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = True
    # every_new_building_providers_scale = 10
    # # 总规模
    # E_seting = 20   #要求E_seting与every_new_building_providers_scale之间呈整数倍关系；
    #
    # # 测试4：已建服务设施不调整，非固定provider count，非固定 provider scale
    # # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = False
    # new_building_providers_count_fixed = None  # 无效的变量，不会被用到的
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = False
    # every_new_building_providers_scale = None   # 无效的变量，不会被用到的
    # # 总规模
    # E_seting = 20
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_numpy_2(0,
                                                                                                           DEMANDS_COUNT,
                                                                                                           OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  # 必须要先创建索引，才可以执行
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    nsga2_obj.excute(demands_provider_np, demands_pdd_np)


def main_2():
    global provider_id_list
    # 参数设置  代
    N_GENERATIONS2 = 10  # 迭代次数
    # 区
    POP_SIZE2 = 1000  # 种群大小
    pc2 = 0.25  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 3
    DEMANDS_COUNT = 9  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 14
    OLD_PROVIDERS_COUNT = 2
    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 10000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 20  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory"  # COLLECTION 名称

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_numpy_2(0,
                                                                                                           DEMANDS_COUNT,
                                                                                                           OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  # 必须要先创建索引，才可以执行
    # #测试1：已建服务设施不调整，固定provider count，固定 provider scale
    # # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = True
    new_building_providers_count_fixed = 2
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = True
    every_new_building_providers_scale = 10
    # 总规模
    E_seting = 20
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    # 初始化种群
    population_test_1 = nsga2_obj.initial_population()
    # # 测试2：已建服务设施不调整，固定provider count，不固定 provider scale
    # # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = True
    new_building_providers_count_fixed = 2
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = False
    every_new_building_providers_scale = None   #无效的变量，不会被用到的
    # 总规模
    E_seting = 20
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    # 初始化种群
    population_test_2 = nsga2_obj.initial_population()
    #
    # 测试3：已建服务设施不调整，非固定provider count，固定 provider scale
    # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = False
    new_building_providers_count_fixed = None  #无效的变量，不会被用到的
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = True
    every_new_building_providers_scale = 10
    # 总规模
    E_seting = 20   #要求E_seting与every_new_building_providers_scale之间呈整数倍关系；
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    # 初始化种群
    population_test_3 = nsga2_obj.initial_population()
    #
    # # 测试4：已建服务设施不调整，非固定provider count，非固定 provider scale
    # # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = False
    new_building_providers_count_fixed = None  # 无效的变量，不会被用到的
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = False
    every_new_building_providers_scale = None   # 无效的变量，不会被用到的
    # 总规模
    E_seting = 20
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    # 初始化种群
    population_test_4 = nsga2_obj.initial_population()

    # 绘制非劣解的fitness关联图
    # 评价函数
    objectives_fitness_1 = nsga2_obj.fitness(demands_provider_np, demands_pdd_np, population_test_1)
    objectives_fitness_2 = nsga2_obj.fitness(demands_provider_np, demands_pdd_np, population_test_2)
    objectives_fitness_3 = nsga2_obj.fitness(demands_provider_np, demands_pdd_np, population_test_3)
    objectives_fitness_4 = nsga2_obj.fitness(demands_provider_np, demands_pdd_np, population_test_4)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("accessibility")
    ax.set_ylabel("inequality")
    ax.set_zlabel("cover population")
    ax.set_title('Initial Population Fitness')
    type1 = ax.scatter(objectives_fitness_1[:, 0], objectives_fitness_1[:, 1], objectives_fitness_1[:, 2], c='r' ,marker='x', s=300)
    type2 = ax.scatter(objectives_fitness_2[:, 0], objectives_fitness_2[:, 1], objectives_fitness_2[:, 2], c='g' ,marker='*', s=80)
    type3 = ax.scatter(objectives_fitness_3[:, 0], objectives_fitness_3[:, 1], objectives_fitness_3[:, 2], c='b' ,marker='+', s=300)
    type4 = ax.scatter(objectives_fitness_4[:, 0], objectives_fitness_4[:, 1], objectives_fitness_4[:, 2], c='grey' ,marker='o', s=50)

    # plt.legend((type1), (u'Non-dominated solution'))
    plt.show()

    df1 = pd.DataFrame(objectives_fitness_1)
    df1.loc[:,"type"]="type1"
    df2 = pd.DataFrame(objectives_fitness_2)
    df2.loc[:, "type"] = "type2"
    df3 = pd.DataFrame(objectives_fitness_3)
    df3.loc[:, "type"] = "type3"
    df4 = pd.DataFrame(objectives_fitness_4)
    df4.loc[:, "type"] = "type4"
    df=pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
    df.columns = ['accessibility', 'inequality', 'cover_population',"type"]
    sns.pairplot(df, hue='type', height=2.5,markers=["x", "*", "+","o"]);
    plt.show()


def main_1919():
    """
    19*19为361，其中100个为demands，Providers为261个，其中已建Provider为4个，待建4个
    :return:
    """
    global provider_id_list
    # 参数设置  代
    N_GENERATIONS2 = 500  # 迭代次数
    # 区
    POP_SIZE2 = 1000  # 种群大小
    pc2 = 0.25  # 交叉概率
    pm2 = 0.25  # 变异概率
    f_num = 3
    DEMANDS_COUNT = 100  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 261-4
    OLD_PROVIDERS_COUNT = 4
    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 100000  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_theory19"  # COLLECTION 名称

    # #测试1：已建服务设施不调整，固定provider count，固定 provider scale
    # # 已建服务设施是否调整
    is_builded_providers_adjust = False
    # 是否新建固定数量的provider
    is_new_building_providers_count_fixed = True
    new_building_providers_count_fixed = 4
    # 每个provider的规模是否相同
    is_every_new_building_providers_scale_equal = True
    every_new_building_providers_scale = 10
    # 总规模
    E_seting = 40

    # # 测试2：已建服务设施不调整，固定provider count，不固定 provider scale
    # # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = True
    # new_building_providers_count_fixed = 2
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = False
    # every_new_building_providers_scale = None   #无效的变量，不会被用到的
    # # 总规模
    # E_seting = 20
    #
    # 测试3：已建服务设施不调整，非固定provider count，固定 provider scale
    # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = False
    # new_building_providers_count_fixed = None  #无效的变量，不会被用到的
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = True
    # every_new_building_providers_scale = 10
    # # 总规模
    # E_seting = 20   #要求E_seting与every_new_building_providers_scale之间呈整数倍关系；
    #
    # # 测试4：已建服务设施不调整，非固定provider count，非固定 provider scale
    # # 已建服务设施是否调整
    # is_builded_providers_adjust = False
    # # 是否新建固定数量的provider
    # is_new_building_providers_count_fixed = False
    # new_building_providers_count_fixed = None  # 无效的变量，不会被用到的
    # # 每个provider的规模是否相同
    # is_every_new_building_providers_scale_equal = False
    # every_new_building_providers_scale = None   # 无效的变量，不会被用到的
    # # 总规模
    # E_seting = 20
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_numpy_2(0,
                                                                                                           DEMANDS_COUNT,
                                                                                                           OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  # 必须要先创建索引，才可以执行
    # 定义NSGA和执行主程序
    nsga2_obj = NSGA2(pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2,
                      N_GENERATIONS2, f_num,
                      is_builded_providers_adjust, is_new_building_providers_count_fixed,
                      new_building_providers_count_fixed, is_every_new_building_providers_scale_equal,
                      every_new_building_providers_scale, E_seting)
    nsga2_obj.excute(demands_provider_np, demands_pdd_np)
# NSGA2入口
if __name__ == '__main__':
    main_1919()

