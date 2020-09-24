"""
参考地址：https://blog.csdn.net/qq_36449201/article/details/81046586
作者：华电小炸扎
"""

import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import a_mongo_operater_theory



class NSGA2():
    def __init__(self, pc, pm, low, up, old_providers_count,  THROD, BEITA, x_dim, pop_size, max_iter):  # 维度，群体数量，迭代次数
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


    def initial_population(self):  # 初始化种群
        self.population = np.zeros(shape=(self.pop_size, self.x_dim))
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                self.population[i][j] = random.random()
        print("initial population completed.")

    # 方式1：定规模、定个数
    def initial_population_with_scale_count(self):
        population = np.full(shape=(self.pop_size, self.x_dim), fill_value=0)
        for m in range(self.pop_size):
            index_1 = np.random.randint(low=0, high=self.x_dim)
            index_2 = np.random.randint(low=0, high=self.x_dim)
            while (index_1 == index_2):
                index_2 = np.random.randint(low=0, high=self.x_dim)
            creating_mark = np.full(shape=(self.x_dim), fill_value=0)
            creating_mark[index_1] = 1
            creating_mark[index_2] = 1  # 数量是建2个
            population[m, :] = creating_mark * 10  # 规模都是10
            if np.argwhere(population[m, :] == 10).flatten().shape[0] < 2:
                print("test")
        return population

    def conbine_children_parent(self,population,new_population_from_selection_mutation):  # 父代种群和子代种群合并,pop*2
        population_child_conbine = np.zeros((2 * self.pop_size, self.x_dim))  # self.population
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                population_child_conbine[i][j] = population[i][j]
                population_child_conbine[i + self.pop_size][j] = new_population_from_selection_mutation[i][j]
        return population_child_conbine



    def select_population_from_parent(self, population_child_conbine, fronts, crowding_distance):  # 根据排序和拥挤度计算，选取新的父代种群 pop*2 到 pop*1
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
                    if i>=len(fronts):
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

        ##############################################begin:fitness#########################################################

        # fitness计算
    def fitness(self, demands_provider_np, demands_pdd_np, popu):
        """
        全局的适应度函数
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


    def calculate_global_waiting_time_numpy(self, demands_np, solution):
        """
        计算全局等候时间
        :param demands_np:
        :param solution:
        :return:
        """
        # ？？？？？？？？？？？？？？？？？？？？
        return 0

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
        # print("test")

    ##############################################end:fitness#########################################################

    def fast_non_dominated_sort(self,objectives_fitness):
        y1_values=objectives_fitness[:,0]
        y2_values=objectives_fitness[:,1]
        y3_values=objectives_fitness[:,2]
        S = [[] for i in range(0, np.shape(y1_values)[0])]
        fronts = [[]]
        n = [0 for i in range(0, np.shape(y1_values)[0])]
        rank = [0 for i in range(0, np.shape(y1_values)[0])]

        for p in range(0, np.shape(y1_values)[0]):
            S[p] = []
            n[p] = 0
            for q in range(0, np.shape(y1_values)[0]):
                # 这是目标函数，y1求小值，y2求小值
                if (y1_values[p] <= y1_values[q] and y2_values[p] <= y2_values[q] and y3_values[p] <= y3_values[q]):
                    if q not in S[p]:
                        # 个体p的支配集合Sp计算
                        S[p].append(q)
                elif (y1_values[p] > y1_values[q] and y2_values[p] > y2_values[q] and y3_values[p] > y3_values[q]):
                    # 被支配度Np计算
                    # Np越大，则说明p个体越差
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in fronts[0]:
                    fronts[0].append(p)

        i = 0
        while (fronts[i] != []):
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            fronts.append(Q)

        del fronts[len(fronts) - 1]
        sum_coun = 0
        for kk in range(len(fronts)):
            sum_coun += len(fronts[kk])
        print(sum_coun)
        return fronts

    def non_donminate(self,population,objectives_fitness):  # pop*2行
        fronts = []  # Pareto前沿面
        fronts.append([])
        set_sp = []
        npp = np.zeros(population.shape[0])
        rank = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            temp = []
            for j in range(population.shape[0]):
                # temp=[]
                if j != i:
                    if objectives_fitness[i][0] >= objectives_fitness[j][0] and objectives_fitness[i][1] >= objectives_fitness[j][1]and objectives_fitness[i][2] >= objectives_fitness[j][2]:
                        npp[i] += 1  # j支配 i，np+1
                    if objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]and objectives_fitness[j][2] >= objectives_fitness[i][2]:
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
        return fronts[0:-1]


    def selection(self, population, demands_provider_np, demands_pdd_np):  # 轮盘赌选择
        fronts, objectives_fitness=self.non_donminate(population, demands_provider_np, demands_pdd_np)  # 非支配排序,获得Pareto前沿面
        pi = np.zeros(self.pop_size)  # 个体的概率
        qi = np.zeros(self.pop_size + 1)  # 个体的累积概率
        P = 0
        for i in range(len(fronts)):
            # for j in range(len(self.fronts[i])):
            P += (1 / (i + 1)) * (len(fronts[i]))  # 累积适应度
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                pi[fronts[i][j]] = (1 / (i + 1)) / P  # 个体遗传到下一代的概率
        for i in range(self.pop_size):
            qi[0] = 0
            qi[i + 1] = np.sum(pi[0:i + 1])  # 累积概率
        new_population_from_selection_mutation = np.zeros((self.pop_size, self.x_dim))
        for i in range(self.pop_size):
            r = random.random()  # 生成随机数，
            a = 0
            for j in range(self.pop_size):
                if r > qi[j] and r < qi[j + 1]:
                    while a < self.x_dim:
                        new_population_from_selection_mutation[i][a] =population[j][a]
                        a += 1
                j += 1
        return new_population_from_selection_mutation

    def crossover(self,new_population_from_selection_mutation):  # 交叉,SBX交叉
        for i in range(self.pop_size - 1):
            # temp1 = []
            # temp2 = []
            if random.random() < self.pc:
                # pc_point = random.randint(0,self.dim-1)        #生成交叉点
                # temp1.append(self.population[i][pc_point:self.dim])
                # temp2.append(self.population[i+1][pc_point:self.dim])
                # self.population[i][pc_point:self.dim] = temp2
                # self.population[i+1][pc_point:self.dim] = temp1
                a = random.random()
                for j in range(self.x_dim):
                    new_population_from_selection_mutation[i][j] = a * new_population_from_selection_mutation[i][j] + (1 - a) * new_population_from_selection_mutation[i + 1][j]
                    new_population_from_selection_mutation[i + 1][j] = a * new_population_from_selection_mutation[i + 1][j] + (1 - a) * new_population_from_selection_mutation[i][j]
            i += 2
        return new_population_from_selection_mutation

    def mutation(self,new_population_from_selection_mutation):  # 变异
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                if random.random() < self.pm:
                    new_population_from_selection_mutation[i][j] = new_population_from_selection_mutation[i][j] - 0.1 + np.random.random() * 0.2
                    if new_population_from_selection_mutation[i][j] < 0:
                        new_population_from_selection_mutation[i][j] = 0  # 最小值0
                    if new_population_from_selection_mutation[i][j] > 1:
                        new_population_from_selection_mutation[i][j] = 1  # 最大值1
        return new_population_from_selection_mutation

    def sample_index_with_scale_count(self,solution,count=1):
        effective_index=np.argwhere(solution == 10).flatten()
        if effective_index.shape[0]==0:
            print("test")
        # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
        solution_a_index_1 = np.random.randint(low=0, high=effective_index.shape[0])
        return effective_index[solution_a_index_1]

    # Function to carry out the crossover
    def crossover_with_scale_count(self, solution_a, solution_b, pc):
        crossover_prob = random.random()
        if crossover_prob < pc:
            solution_a_index_1 =self.sample_index_with_scale_count(solution_a,count=1)
            solution_b_index_1 = self.sample_index_with_scale_count(solution_b,count=1)
            if solution_a_index_1 != solution_b_index_1:
                if solution_a[solution_b_index_1]!=10:
                    solution_a[solution_b_index_1]=10
                    solution_a[solution_a_index_1]= 0
            if np.argwhere(solution_a == 10).flatten().shape[0] < 2:
                print("test")
        return solution_a

    # Function to carry out the mutation operator
    def mutation_with_scale_count(self, solution, pm):
        mutation_prob = random.random()
        if mutation_prob < pm:
            index_1 = np.random.randint(low=0, high=self.x_dim)
            index_2 = np.random.randint(low=0, high=self.x_dim)
            while (index_1 == index_2):
                index_2 = np.random.randint(low=0, high=self.x_dim)
            creating_mark = np.full(shape=(self.x_dim), fill_value=0)
            creating_mark[index_1] = 1
            creating_mark[index_2] = 1  # 数量是建2个
            solution = creating_mark * 10  # 规模都是10
        return solution

    # 方式1：定规模、定数量   2个，每个建10
    def crossover_and_mutation_simple_with_scale_count(self, popu, pc, pm):
        off_spring=np.full(shape=(popu.shape[0], self.x_dim), fill_value=0)
        for i in range(popu.shape[0]):
            a1 = random.randint(0, popu.shape[0]-1)
            b1 = random.randint(0, popu.shape[0]-1)
            # 通过crossover和mutation的方式生成新的个体
            solution=self.crossover_with_scale_count(popu[a1,:], popu[b1,:],pc)
            if np.argwhere(solution  == 10).flatten().shape[0] < 2:
                print("test")
            off_spring[i,:]=self.mutation_with_scale_count(solution,pm)
            if np.argwhere(off_spring[i,:]  == 10).flatten().shape[0] < 2:
                print("test")
        return off_spring

    def crowd_distance(self,fronts,objectives_fitness):  # 拥挤度计算，前沿面每个个体的拥挤度
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
                crowding_distance[int(temp1[a][1])] = (temp1[a + 1][0] - temp1[a - 1][0]) / (f1_max - f1_min) + \
                                                      (temp2[a + 1][0] - temp2[a - 1][0]) / (f2_max - f2_min) + \
                                                      (temp3[a + 1][0] - temp3[a - 1][0]) / (f3_max - f3_min) # 个体i的拥挤度等于 f1 + f2 + f3
                a += 1
        return crowding_distance

    def draw(self,demands_provider_np, demands_pdd_np,population):  # 画图
        # 评价函数
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population)
        # fronts=self.non_donminate(population,objectives_fitness)
        fronts=self.fast_non_dominated_sort(objectives_fitness)
        objectives_fitness = self.fitness(demands_provider_np, demands_pdd_np, population[fronts[0]])

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
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('ZDT3 Pareto Front')
        plt.grid()
        plt.show()
        # plt.savefig('nsga2 ZDT2 Pareto Front 2.png')

        #将值输出出来
        for iii in range(len(population[fronts[0]])):
            print(population[fronts[0]][iii,:])
        return population[fronts[0]]

    def excute(self, demands_provider_np, demands_pdd_np):  # 主程序
        # 初始化种群
        population=self.initial_population_with_scale_count()
        for i in range(self.max_iteration):
            #选择、交叉、变异，生成子代种群
            matingpool = random.sample(range(self.pop_size), self.pop_size)
            new_population_from_selection_mutation = self.crossover_and_mutation_simple_with_scale_count(population[matingpool, :], self.pc,self.pm)  # 遗传算子,模拟二进制交叉和多项式变异
            # new_population_from_selection_mutation=self.selection(population,demands_provider_np, demands_pdd_np)
            # new_population_from_selection_mutation=self.crossover(new_population_from_selection_mutation)
            # new_population_from_selection_mutation=self.mutation(new_population_from_selection_mutation)
            # 父代与子代种群合并
            population_child_conbine = self.conbine_children_parent(population,new_population_from_selection_mutation)
            # 评价函数
            objectives_fitness = self.fitness(demands_provider_np,demands_pdd_np,population_child_conbine)
            # 快速非支配排序
            # fronts=self.non_donminate(population_child_conbine,objectives_fitness)
            fronts=self.fast_non_dominated_sort(objectives_fitness)
            # 拥挤度计算
            crowding_distance=self.crowd_distance(fronts,objectives_fitness)
            # 根据Pareto等级和拥挤度选取新的父代种群，选择交叉变异
            population=self.select_population_from_parent(population_child_conbine, fronts, crowding_distance)
            print(i,"代")
        result_population=self.draw(demands_provider_np, demands_pdd_np,population)

        # print(self.fronts)
        # print(self.population)
        # print(self.new_popu)
        # print(self.popu_child)
        # print(self.objectives)
        # print()
        return result_population

# NSGA2入口
if __name__ == '__main__':
    # 参数设置  代
    N_GENERATIONS2 = 200  # 迭代次数
    # 区
    POP_SIZE2 = 200  # 种群大小
    pc2 = 0.25  # 交叉概率
    pm2 = 0.25  # 变异概率

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
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list = mongo_operater_obj.find_records_format_numpy_2(0,
                                                                                                           DEMANDS_COUNT,
                                                                                                           OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  # 必须要先创建索引，才可以执行
    NSGA = NSGA2( pc2, pm2, low2, up2, OLD_PROVIDERS_COUNT, THROD, BEITA, PENTENTIAL_NEW_PROVIDERS_COUNT, POP_SIZE2, N_GENERATIONS2)
    result_popu=NSGA.excute(demands_provider_np, demands_pdd_np)

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
    import  e_03_naga3_ps_change_DS_theory
    e_03_naga3_ps_change_DS_theory.test_frequency_show(result_provider_id_popu)