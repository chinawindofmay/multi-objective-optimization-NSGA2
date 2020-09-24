"""
参考地址：https://blog.csdn.net/qq_36449201/article/details/81046586
作者：华电小炸扎
"""

import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt



class NSGA2():
    def __init__(self, x_dim, pop_size, max_iter):  # 维度，群体数量，迭代次数
        self.pc = 0.25  # 交叉概率
        self.pm = 0.25  # 变异概率
        self.x_dim = x_dim  # 搜索维度
        self.pop_size = pop_size  # 总群个体数量
        self.max_iteration = max_iter  # 迭代次数
        self.population = []  # 父代种群
        self.new_population_from_selection_mutation = []  # 选择算子、变异算子操作过后的新种群
        # self.children = []                     #子代种群
        self.population_child_conbine = []  # 合并后的父代与子代种群
        self.fronts = []  # Pareto前沿面
        self.rank = []  # np.zeros(self.pop)       #非支配排序等级
        self.crowding_distance = []  # 个体拥挤度
        self.objectives_fitness = []  # 目标函数值,pop行 2列
        self.set_sp = []  # 个体 i的支配解集
        self.np = []  # 该个体被支配的数目

    def initial_population(self):  # 初始化种群
        self.population = np.zeros(shape=(self.pop_size, self.x_dim))
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                self.population[i][j] = random.random()
        print("initial population completed.")

    def conbine_children_parent(self):  # 父代种群和子代种群合并,pop*2
        self.population_child_conbine = np.zeros((2 * self.pop_size, self.x_dim))  # self.population
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                self.population_child_conbine[i][j] = self.population[i][j]
                self.population_child_conbine[i + self.pop_size][j] = self.new_population_from_selection_mutation[i][j]

    def select_newparent(self):  # 根据排序和拥挤度计算，选取新的父代种群 pop*2 到 pop*1
        self.population = np.zeros((self.pop_size, self.x_dim))  # 选取新的种群
        try:
            a = len(self.fronts[0])  # Pareto前沿面第一层 个体的个数
            if a >= self.pop_size:
                for i in range(self.pop_size):
                    self.population[i] = self.population_child_conbine[self.fronts[0][i]]
            else:
                d = []  # 用于存放前b层个体
                i = 1
                while a < self.pop_size:
                    # if i>=len(self.fronts):
                    #     print("test")
                    #     break
                    c = a  # 新种群内 已经存放的个体数目    *列
                    a += len(self.fronts[i])
                    for j in range(len(self.fronts[i - 1])):
                        d.append(self.fronts[i - 1][j])
                        # while d < self.dim:
                        # self.population[j][d] = self.popu_child[self.fronts[i-1][j]][d]
                        # d += 1
                    b = i  # 第b层不能放，超过种群数目了    *行
                    i = i + 1
                # 把前c个放进去
                for j in range(c):
                    self.population[j] = self.population_child_conbine[d[j]]
                temp = np.zeros((len(self.fronts[b]), 2))  # 存放拥挤度和个体序号
                for i in range(len(self.fronts[b])):
                    temp[i][0] = self.crowding_distance[self.fronts[b][i]]
                    temp[i][1] = self.fronts[b][i]
                temp = sorted(temp.tolist())  # 拥挤距离由小到大排序
                for i in range(self.pop_size - c):
                    # 按拥挤距离由大到小填充直到种群数量达到 pop
                    self.population[c + i] = self.population_child_conbine[int(temp[len(temp) - i - 1][1])]
        except Exception as E_results:
            print("捕捉有异常：", E_results)
            self.population = self.population_child_conbine[0:self.pop_size, :]


    def cal_obj_ZDT2(self, position):  # 计算一个个体的多目标函数值 f1,f2 最小值
        f1 = position[0]
        f = 0
        for i in range(self.x_dim - 1):
            f += 9 * (position[i + 1] / (self.x_dim - 1))
        g = 1 + f
        f2 = g * (1 - np.square(f1 / g))
        return [f1, f2]

    def cal_obj_ZDT3(self,position):
        f1=position[0]
        sum1 = np.array(np.sum(position[1:]))
        g = (1 + 9 * sum1 / 29).astype(np.float)
        h = 1 - np.power(f1 / g, 0.5) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return [f1,f2]

    def cal_obj_ZDT4(self,position):
        f1 = position[0]
        sum1 = np.array(np.sum(np.power(position[1:], 2) - 10 * np.cos(4 * np.pi * position[1:])))
        g = (91 + sum1).astype(np.float)
        f2 = (g * (1 - (f1 / g) ** 0.5)).astype(np.float)
        return [f1,f2]

    def cal_obj_ZDT6(self,position):
        f1 = (1 - np.exp(-4 * position[0]) * (np.sin(6 * np.pi * position[0])) ** 6).astype( np.float)
        sum1 = np.array(np.sum(position[1:]) )
        g = (1 + 9 * ((sum1 / (10 - 1)) ** 0.25)).astype(np.float)
        f1 = (1 - np.exp(-4 * position[0]) * (np.sin(6 * np.pi * position[0])) ** 6).astype(np.float)
        f2 = g * (1 - (f1 / g) ** 2)
        return [f1,f2]

    def non_donminate2(self):  # pop*2行
        self.fronts = []  # Pareto前沿面
        self.fronts.append([])
        self.set_sp = []
        self.objectives_fitness = []  # np.zeros((2*self.pop,2))
        self.np = np.zeros(2 * self.pop_size)
        self.rank = np.zeros(2 * self.pop_size)
        position = []
        for i in range(2 * self.pop_size):  # 越界处理
            for j in range(self.x_dim):
                if self.population_child_conbine[i][j] < 0:
                    self.population_child_conbine[i][j] = 0  # 最小值0
                if self.population_child_conbine[i][j] > 1:
                    self.population_child_conbine[i][j] = 1  # 最大值1
        for i in range(2 * self.pop_size):
            position = self.population_child_conbine[i]
            # self.cal_obj(position)
            self.objectives_fitness.append(self.cal_obj_ZDT6(position))  # [i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
            # self.objectives[i][1] = f2
        for i in range(2 * self.pop_size):
            temp = []
            for j in range(2 * self.pop_size):
                # temp=[]
                if j != i:
                    if (self.objectives_fitness[i][0] >= self.objectives_fitness[j][0] and self.objectives_fitness[i][1] >= self.objectives_fitness[j][1]) or \
                        (self.objectives_fitness[i][0] > self.objectives_fitness[j][0] and self.objectives_fitness[i][1] >= self.objectives_fitness[j][1]) or \
                        (self.objectives_fitness[i][0] >= self.objectives_fitness[j][0] and self.objectives_fitness[i][1] > self.objectives_fitness[j][1]):
                        self.np[i] += 1  # j支配 i，np+1
                    if (self.objectives_fitness[j][0] > self.objectives_fitness[i][0] and self.objectives_fitness[j][1] > self.objectives_fitness[i][1]) or \
                        (self.objectives_fitness[j][0] > self.objectives_fitness[i][0] and self.objectives_fitness[j][1] >= self.objectives_fitness[i][1]) or \
                        (self.objectives_fitness[j][0] >= self.objectives_fitness[i][0] and self.objectives_fitness[j][1] > self.objectives_fitness[i][1]):
                        temp.append(j)
            self.set_sp.append(temp)  # i支配 j，将 j 加入 i 的支配解集里
            if self.np[i] == 0:
                self.fronts[0].append(i)  # 个体序号
                self.rank[i] = 1  # Pareto前沿面 第一层级
        i = 0
        while len(self.fronts[i]) > 0:
            temp = []
            for j in range(len(self.fronts[i])):
                a = 0
                while a < len(self.set_sp[self.fronts[i][j]]):
                    self.np[self.set_sp[self.fronts[i][j]][a]] -= 1
                    if self.np[self.set_sp[self.fronts[i][j]][a]] == 0:
                        self.rank[self.set_sp[self.fronts[i][j]][a]] = i + 2  # 第二层级
                        temp.append(self.set_sp[self.fronts[i][j]][a])
                    a = a + 1
            i = i + 1
            self.fronts.append(temp)
        # self.fronts=self.fronts[0:-1]
        # test code
        sum_coun = 0
        for kk in range(len(self.fronts)):
            sum_coun += len(self.fronts[kk])
        if sum_coun==20:
            print("test")


    def non_donminate_1(self):  # pop行 快速非支配排序
        self.fronts = []  # Pareto前沿面
        self.fronts.append([])
        self.set_sp = []
        self.objectives_fitness = []  # np.zeros((self.pop,2))
        self.np = np.zeros(self.pop_size)
        self.rank = np.zeros(self.pop_size)
        position = []
        for i in range(self.pop_size):  # 越界处理
            for j in range(self.x_dim):
                if self.population[i][j] < 0:
                    self.population[i][j] = 0  # 最小值0
                if self.population[i][j] > 1:
                    self.population[i][j] = 1  # 最大值1
        for i in range(self.pop_size):
            position = self.population[i]
            # self.cal_obj(position)
            self.objectives_fitness.append(self.cal_obj_ZDT6(position))  # [i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
            # self.objectives[i][1] = f2
        for i in range(self.pop_size):
            temp = []
            for j in range(self.pop_size):
                # temp=[]
                if j != i:
                    if self.objectives_fitness[i][0] >= self.objectives_fitness[j][0] and self.objectives_fitness[i][1] >= self.objectives_fitness[j][
                        1]:
                        self.np[i] += 1  # j支配 i，np+1
                    if self.objectives_fitness[j][0] >= self.objectives_fitness[i][0] and self.objectives_fitness[j][1] >= self.objectives_fitness[i][
                        1]:
                        temp.append(j)
            self.set_sp.append(temp)  # i支配 j，将 j 加入 i 的支配解集里
            if self.np[i] == 0:
                self.fronts[0].append(i)  # 个体序号
                self.rank[i] = 1  # Pareto前沿面 第一层级
        i = 0
        while len(self.fronts[i]) > 0:
            temp = []
            for j in range(len(self.fronts[i])):
                a = 0
                while a < len(self.set_sp[self.fronts[i][j]]):
                    self.np[self.set_sp[self.fronts[i][j]][a]] -= 1
                    if self.np[self.set_sp[self.fronts[i][j]][a]] == 0:
                        self.rank[self.set_sp[self.fronts[i][j]][a]] = i + 2  # 第二层级
                        temp.append(self.set_sp[self.fronts[i][j]][a])
                    a = a + 1
            i = i + 1
            self.fronts.append(temp)
        # self.fronts=self.fronts[0:-1]


    def selection(self):  # 轮盘赌选择
        self.non_donminate_1()  # 非支配排序,获得Pareto前沿面
        pi = np.zeros(self.pop_size)  # 个体的概率
        qi = np.zeros(self.pop_size + 1)  # 个体的累积概率
        P = 0
        for i in range(len(self.fronts)):
            # for j in range(len(self.fronts[i])):
            P += (1 / (i + 1)) * (len(self.fronts[i]))  # 累积适应度
        for i in range(len(self.fronts)):
            for j in range(len(self.fronts[i])):
                pi[self.fronts[i][j]] = (1 / (i + 1)) / P  # 个体遗传到下一代的概率
        for i in range(self.pop_size):
            qi[0] = 0
            qi[i + 1] = np.sum(pi[0:i + 1])  # 累积概率
        self.new_population_from_selection_mutation = np.zeros((self.pop_size, self.x_dim))
        for i in range(self.pop_size):
            r = random.random()  # 生成随机数，
            a = 0
            for j in range(self.pop_size):
                if r > qi[j] and r < qi[j + 1]:
                    while a < self.x_dim:
                        self.new_population_from_selection_mutation[i][a] = self.population[j][a]
                        a += 1
                j += 1

    def crossover(self):  # 交叉,SBX交叉
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
                    self.new_population_from_selection_mutation[i][j] = a * self.new_population_from_selection_mutation[i][j] + (1 - a) * self.new_population_from_selection_mutation[i + 1][j]
                    self.new_population_from_selection_mutation[i + 1][j] = a * self.new_population_from_selection_mutation[i + 1][j] + (1 - a) * self.new_population_from_selection_mutation[i][j]
            i += 2

    def mutation(self):  # 变异
        for i in range(self.pop_size):
            for j in range(self.x_dim):
                if random.random() < self.pm:
                    self.new_population_from_selection_mutation[i][j] = self.new_population_from_selection_mutation[i][j] - 0.1 + np.random.random() * 0.2
                    if self.new_population_from_selection_mutation[i][j] < 0:
                        self.new_population_from_selection_mutation[i][j] = 0  # 最小值0
                    if self.new_population_from_selection_mutation[i][j] > 1:
                        self.new_population_from_selection_mutation[i][j] = 1  # 最大值1

    def crowd_distance(self):  # 拥挤度计算，前沿面每个个体的拥挤度
        self.crowding_distance = np.zeros(2 * self.pop_size)
        for i in range(len(self.fronts) - 1):  # fronts最后一行为空集
            temp1 = np.zeros((len(self.fronts[i]), 2))
            temp2 = np.zeros((len(self.fronts[i]), 2))
            for j in range(len(self.fronts[i])):
                temp1[j][0] = self.objectives_fitness[self.fronts[i][j]][0]  # f1赋值
                temp1[j][1] = self.fronts[i][j]
                temp2[j][0] = self.objectives_fitness[self.fronts[i][j]][1]  # f2赋值
                temp2[j][1] = self.fronts[i][j]
            # temp3 = temp1.tolist()
            # temp4 = temp2.tolist()
            temp1 = sorted(temp1.tolist())  # f1排序
            temp2 = sorted(temp2.tolist())  # f2排序
            self.crowding_distance[int(temp1[0][1])] = float('inf')
            self.crowding_distance[int(temp1[len(self.fronts[i]) - 1][1])] = float('inf')
            f1_min = temp1[0][0]
            f1_max = temp1[len(self.fronts[i]) - 1][0]
            f2_max = temp2[len(self.fronts[i]) - 1][0]
            f2_min = temp2[0][0]
            a = 1
            while a < len(self.fronts[i]) - 1:
                self.crowding_distance[int(temp1[a][1])] = (temp1[a + 1][0] - temp1[a - 1][0]) / (f1_max - f1_min) + (
                            temp2[a + 1][0] - temp2[a - 1][0]) / (f2_max - f2_min)  # 个体i的拥挤度等于 f1 + f2
                a += 1

    def draw(self):  # 画图
        self.objectives_fitness = []  # np.zeros((self.pop,2))
        position = []
        for i in range(self.pop_size):  # 越界处理
            for j in range(self.x_dim):
                if self.population[i][j] < 0:
                    self.population[i][j] = 0  # 最小值0
                if self.population[i][j] > 1:
                    self.population[i][j] = 1  # 最大值1
        self.non_donminate_1()
        for i in range(len(self.fronts[0])):
            position = self.population[self.fronts[0][i]]
            self.objectives_fitness.append(self.cal_obj_ZDT6(position))
        # for i in range(self.pop):
        # position = self.population[i]
        # self.objectives.append(self.cal_obj(position))#[i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
        x = []
        y = []
        for i in range(self.pop_size):
            x.append(self.objectives_fitness[i][0])
            y.append(self.objectives_fitness[i][1])
        ax = plt.subplot(111)
        plt.scatter(x, y)
        # plt.plot(,'--',label='')
        plt.axis([0.0, 1.0, -1, 5])
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

    def run(self):  # 主程序
        # 初始化种群
        self.initial_population()
        for i in range(self.max_iteration):
            #选择、交叉、变异，生成子代种群
            self.selection()
            self.crossover()
            self.mutation()
            # 父代与子代种群合并
            self.conbine_children_parent()
            # 快速非支配排序
            self.non_donminate2()

            # 拥挤度计算
            self.crowd_distance()
            # 根据Pareto等级和拥挤度选取新的父代种群，选择交叉变异
            self.select_newparent()
            print(i,"代")
        self.draw()
        # print(self.fronts)
        # print(self.population)
        # print(self.new_popu)
        # print(self.popu_child)
        # print(self.objectives)
        # print()


def fast_non_dominated_sort(y1_values, y2_values):
    S=[[] for i in range(0, np.shape(y1_values)[0])]
    fronts = [[]]
    n=[0 for i in range(0, np.shape(y1_values)[0])]
    rank = [0 for i in range(0, np.shape(y1_values)[0])]

    for p in range(0, np.shape(y1_values)[0]):
        S[p]=[]
        n[p]=0
        for q in range(0, np.shape(y1_values)[0]):
            # 这是目标函数，y1求小值，y2求小值
            if (y1_values[p] < y1_values[q] and y2_values[p] < y2_values[q] )  :
                if q not in S[p]:
                    # 个体p的支配集合Sp计算
                    S[p].append(q)
            elif (y1_values[p] > y1_values[q] and y2_values[p] > y2_values[q] ) :
                # 被支配度Np计算
                # Np越大，则说明p个体越差
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in fronts[0]:
                fronts[0].append(p)

    i = 0
    while(fronts[i] != []):
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        fronts.append(Q)

    del fronts[len(fronts)-1]
    sum_coun = 0
    for kk in range(len(fronts)):
        sum_coun += len(fronts[kk])
    print(sum_coun)
    return fronts

def test_no_sort():
    fronts = []  # Pareto前沿面
    fronts.append([])
    set_sp = []
    npp = np.zeros(2*10)
    rank = np.zeros(2*10)
    objectives_fitness=np.array([[0.6373723585181119, 9.089424920752537],
                                 [0.6307563745957109, 9.484134522661321],
                                 [0.6307726564027054, 9.370453232401315],
                                 [0.9214017573731662, 8.974173267351892],
                                 [0.8106208092655269, 9.00814519794432],
                                 [0.6308299859236132, 9.381311843663337],
                                 [0.9996933421004693, 8.998732212375378],
                                 [0.8106208092655269, 9.087298161567794],
                                 [0.9968206553777186, 9.020037858133483],
                                 [0.9503113437004427, 9.04519027298749],
                                 [0.9214017573731662, 8.974173267351892],
                                 [0.9214017573731662, 9.00187266065025],
                                 [0.6307563745957109, 9.493829448943414],
                                 [0.999999999999489, 9.180616877016963],
                                 [0.8150090640161689, 9.041622216379706],
                                 [0.9990805452551389, 8.980910429010232],
                                 [0.9979468094812165, 9.04561922020985],
                                 [0.7527617476769539, 9.136335451211739],
                                 [0.6364241984468356, 9.20992553291975],
                                 [0.8106208092655269, 9.083868693443087]])

    objectives_fitness_20=np.array([[0.9876243312911611, 10.838098629526387],
                                 [0.991433893626072, 11.175720619772605],
                                 [0.9870262883942262, 10.920867921752558],
                                 [0.9529952871492675, 11.042348488543286],
                                 [0.9967715948628068, 10.792857629818656],
                                 [0.2914471050280927, 10.71215723744884],
                                 [0.9989416406019469, 11.121149471714903],
                                 [0.9887065108630387, 10.905419079122366],
                                 [0.9820488305486297, 11.361465712166202],
                                 [0.9996490682834592, 11.26840545995472],
                                 [0.2914471050280927, 10.707453256474697],
                                 [0.991433893626072, 11.176351948132849],
                                 [0.9820488305486297, 11.370681966170503],
                                 [0.9887065108630387, 10.895256565042377],
                                 [0.9870262883942262, 10.900449654701166],
                                 [0.9039846682981595, 11.080658791422916],
                                 [0.9989416406019469, 11.12991202060966],
                                 [0.9876243312911611, 10.814608080295447],
                                 [0.2914471050280927, 10.722602119064524],
                                 [0.9989209883195167, 10.80882164883729]])

    for i in range(2 * 10):
        temp = []
        for j in range(2 * 10):
            if j != i:
                if (objectives_fitness[i][0] >= objectives_fitness[j][0] and objectives_fitness[i][1] >=
                    objectives_fitness[j][1]) :
                    npp[i] += 1  # j支配 i，np+1
                if (objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] >
                    objectives_fitness[i][1]) :
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
    # fronts=fronts[0:-1]
    # test code
    sum_coun = 0
    for kk in range(len(fronts)):
        sum_coun += len(fronts[kk])
    print(sum_coun)

# NSGA2入口
if __name__ == '__main__':
    # NSGA = NSGA2(30, 10, 500)
    # NSGA.run()
    # test_no_sort()

    objectives_fitness = np.array([[0.6373723585181119, 9.089424920752537],
                                   [0.6307563745957109, 9.484134522661321],
                                   [0.6307726564027054, 9.370453232401315],
                                   [0.9214017573731662, 8.974173267351892],
                                   [0.8106208092655269, 9.00814519794432],
                                   [0.6308299859236132, 9.381311843663337],
                                   [0.9996933421004693, 8.998732212375378],
                                   [0.8106208092655269, 9.087298161567794],
                                   [0.9968206553777186, 9.020037858133483],
                                   [0.9503113437004427, 9.04519027298749],
                                   [0.9214017573731662, 8.974173267351892],
                                   [0.9214017573731662, 9.00187266065025],
                                   [0.6307563745957109, 9.493829448943414],
                                   [0.999999999999489, 9.180616877016963],
                                   [0.8150090640161689, 9.041622216379706],
                                   [0.9990805452551389, 8.980910429010232],
                                   [0.9979468094812165, 9.04561922020985],
                                   [0.7527617476769539, 9.136335451211739],
                                   [0.6364241984468356, 9.20992553291975],
                                   [0.8106208092655269, 9.083868693443087]])
    objectives_fitness_20 = np.array([[0.9876243312911611, 10.838098629526387],
                                      [0.991433893626072, 11.175720619772605],
                                      [0.9870262883942262, 10.920867921752558],
                                      [0.9529952871492675, 11.042348488543286],
                                      [0.9967715948628068, 10.792857629818656],
                                      [0.2914471050280927, 10.71215723744884],
                                      [0.9989416406019469, 11.121149471714903],
                                      [0.9887065108630387, 10.905419079122366],
                                      [0.9820488305486297, 11.361465712166202],
                                      [0.9996490682834592, 11.26840545995472],
                                      [0.2914471050280927, 10.707453256474697],
                                      [0.991433893626072, 11.176351948132849],
                                      [0.9820488305486297, 11.370681966170503],
                                      [0.9887065108630387, 10.895256565042377],
                                      [0.9870262883942262, 10.900449654701166],
                                      [0.9039846682981595, 11.080658791422916],
                                      [0.9989416406019469, 11.12991202060966],
                                      [0.9876243312911611, 10.814608080295447],
                                      [0.2914471050280927, 10.722602119064524],
                                      [0.9989209883195167, 10.80882164883729]])
    fast_non_dominated_sort(objectives_fitness[:,0],objectives_fitness[:,1])