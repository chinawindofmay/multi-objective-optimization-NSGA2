#coding:utf-8
# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import NSGA2_05_evaluation as evaluation

####################################################BEGIN:Quantum_NSGA2类####################################################
class Quantum_NSGA2:

    #初始化函数
    def __init__(self,
                 POP_SIZE,
                 MAX_GEN,
                 X_COUNT,
                 ROTATE_PROB_THRESHOLD,
                 MUTATION_PROB_THRESHOLD,
                 MIN_X,
                 MAX_X,
                 DELATE,
                 DISTANCE_INFINTE,
                 fitness1,
                 fitness2,
                 fitness3,
                 CHROMOSOME_LEN,
                 BEGIN_C_M,
                 ANGLE_DATE):
        self.POP_SIZE = POP_SIZE
        self.MAX_GEN = MAX_GEN
        self.X_COUNT = X_COUNT
        self.ROTATE_PROB_THRESHOLD = ROTATE_PROB_THRESHOLD
        self.MUTATION_PROB_THRESHOLD = MUTATION_PROB_THRESHOLD
        # Initialization
        self.MIN_X = MIN_X
        self.MAX_X = MAX_X
        self.DELATE = DELATE
        self.DISTANCE_INFINTE = DISTANCE_INFINTE
        self.fitness1=fitness1
        self.fitness2=fitness2
        self.fitness3=fitness3
        self.CHROMOSOME_LEN=CHROMOSOME_LEN
        self.BEGIN_C_M=BEGIN_C_M
        self.ANGLE_DATE=ANGLE_DATE

    def get_index_of(self,a, list_obj):
        """
        :param a: 元素
        :param list_obj: 列表
        :return: 获取索引,Function to find index of list
        """
        for i in range(0, len(list_obj)):
            if list_obj[i] == a:
                return i
        return -1

    def sort_by_values_min(self,front, y_values):
        """
        :param front:
        :param y_values:
        :return: 按照小值排序的方式，Function to sort by values
        """
        sorted_list = []
        while (len(sorted_list) != len(front)):
            index = np.where(np.min(y_values) == y_values)[0][0]
            if index in front:
                sorted_list.append(index)
            y_values[index] = float(math.inf)
        return sorted_list

    def sort_by_values_max(self,front, y_values):
        """
        :param front:
        :param y_values:
        :return: 按照大值排序的方式，Function to sort by values
        """
        sorted_list = []
        while (len(sorted_list) != len(front)):
            index = np.where(np.max(y_values) == y_values)[0][0]
            if index in front:
                sorted_list.append(index)
            y_values[index] = float(-math.inf)
        return sorted_list

    def fast_non_dominated_sort(self,y1_values, y2_values, y3_values):
        """
        :param y1_values:
        :param y2_values:
        :param y3_values:
        :return: 非支配解排序 Function to carry out NSGA-II's fast non dominated sort
        """
        S = [[] for i in range(0, len(y1_values))]
        fronts = [[]]
        n = [0 for i in range(0, len(y1_values))]
        rank = [0 for i in range(0, len(y1_values))]
        # 求解非支配解
        for p in range(0, len(y1_values)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(y1_values)):
                # 这其中的是目标函数
                # y1求最小，y2求最大，y3求最大
                if (y1_values[p] < y1_values[q] and y2_values[p] > y2_values[q] and y3_values[p] > y3_values[q]):
                    if q not in S[p]:
                        # 个体p的支配集合Sp计算
                        S[p].append(q)
                elif (y1_values[p] > y1_values[q] and y2_values[p] < y2_values[q] and y3_values[p] < y3_values[q]):
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
        return fronts

    def crowding_distance(self,y1_values, y2_values, y3_values, front):
        """
        :param y1_values:
        :param y2_values:
        :param y3_values:
        :param front:
        :return: 拥挤距离计算，Function to calculate crowding distance
        """
        distance_ls = [0 for i in range(0, len(front))]
        # 根据y1的值做一次排序
        sorted1 = self.sort_by_values_min(front, y1_values.copy())
        # 根据y2的值做一次排序
        sorted2 = self.sort_by_values_max(front, y2_values.copy())
        sorted3 = self.sort_by_values_max(front, y3_values.copy())
        # 第一个个体和最后一个个体，定义为无限远
        distance_ls[0] = DISTANCE_INFINTE
        distance_ls[len(front) - 1] = DISTANCE_INFINTE
        distance_y1 = np.max(y1_values) - np.min(y1_values) + DELATE
        distance_y2 = np.max(y2_values) - np.min(y2_values) + DELATE
        distance_y3 = np.max(y3_values) - np.min(y3_values) + DELATE
        # 计算中间个体的距离
        for k in range(1, len(front) - 1):
            distance_ls[k] = distance_ls[k] + (y1_values[sorted1[k + 1]] - y2_values[sorted1[k - 1]]) / distance_y1
            distance_ls[k] = distance_ls[k] + (y1_values[sorted2[k + 1]] - y2_values[sorted2[k - 1]]) / distance_y2
            distance_ls[k] = distance_ls[k] + (y1_values[sorted3[k + 1]] - y2_values[sorted3[k - 1]]) / distance_y3
            distance_ls[k] = distance_ls[k] + (y2_values[sorted1[k + 1]] - y3_values[sorted1[k - 1]]) / distance_y1
            distance_ls[k] = distance_ls[k] + (y2_values[sorted2[k + 1]] - y3_values[sorted2[k - 1]]) / distance_y2
            distance_ls[k] = distance_ls[k] + (y2_values[sorted3[k + 1]] - y3_values[sorted3[k - 1]]) / distance_y3
            distance_ls[k] = distance_ls[k] + (y1_values[sorted1[k + 1]] - y3_values[sorted1[k - 1]]) / distance_y1
            distance_ls[k] = distance_ls[k] + (y1_values[sorted2[k + 1]] - y3_values[sorted2[k - 1]]) / distance_y2
            distance_ls[k] = distance_ls[k] + (y1_values[sorted3[k + 1]] - y3_values[sorted3[k - 1]]) / distance_y3
        return distance_ls

    def crossover_and_mutation(self, chromosome_angle_a, chromosome_angle_b, gen_no):
        """
        :param chromosome_angle_a:
        :param chromosome_angle_b:
        :param gen_no:  当前进化的代数，然后用于改变交叉概率和变异概率
        :return: 交叉，Function to carry out the crossover
        """
        chromosome_angle_a_new=chromosome_angle_a.copy()
        crossover_prob = random.random()
        mutation_prob=random.random()
        if crossover_prob < self.ROTATE_PROB_THRESHOLD*(1-gen_no/MAX_GEN):   #改进为变长概率
            chromosome_angle_a_new[self.BEGIN_C_M:,:]=chromosome_angle_b[self.BEGIN_C_M:,:]
        if mutation_prob<self.MUTATION_PROB_THRESHOLD*(1-gen_no/MAX_GEN):    #改进为变长概率
            chromosome_angle_a_new = self.mutation(chromosome_angle_a_new)  #这里有优化的空间
            #到这里了
        chromosome_qubit_a=np.sin(chromosome_angle_a_new)
        initial_judge = np.random.random(size=(self.CHROMOSOME_LEN, self.X_COUNT))
        chromosome_bit_a_np = np.array(chromosome_qubit_a > initial_judge, np.int)
        ##计算二进制对应的十进制数值
        x_a=[]
        for j in range(self.X_COUNT):
            total_a = int("".join('%s' % id for id in list(chromosome_bit_a_np[:,j])), 2)
            x_1_a = (total_a * (self.MAX_X - self.MIN_X)) / math.pow(2,self.CHROMOSOME_LEN) + self.MIN_X
            x_a.append(x_1_a)
        return x_a,chromosome_angle_a_new

    def mutation(self,chromosome_angle):
        """
        :param chromosome_angle:
        :return: 变异，Function to carry out the mutation operator
        """
        chromosome_angle[self.BEGIN_C_M:,:]=np.pi * 2 * np.random.random(size=(self.CHROMOSOME_LEN-self.BEGIN_C_M,self.X_COUNT))
        return chromosome_angle

    def get_result(self, population):
        """
        :param population:
        :return: 将进化出的结果，进一步提取最优的进行制图表达
        """
        # 输出当前代的最优非支配曲面上的结果
        y1_values = self.fitness1(population[:, 0], population[:, 1])
        y2_values = self.fitness2(population[:, 0], population[:, 1])
        y3_values = self.fitness3(population[:, 0], population[:, 1])
        #非支配排序，non_dominated_sorted_fronts[0]为最优支配曲面
        non_dominated_sorted_fronts = self.fast_non_dominated_sort(y1_values[:], y2_values[:], y3_values[:])
        len_non_dominated_sorted_fronts = len(non_dominated_sorted_fronts[0])
        pof_population = np.random.uniform(MIN_X, MAX_X, (len_non_dominated_sorted_fronts, X_COUNT))  # 存放x1,x2
        # 获取POF，最优非支配曲面(Pareto - optimal front)
        pof_i = 0
        for index_x in non_dominated_sorted_fronts[0]:
            print("x={0},y1={1},y2={2},y3={3}".format(population[index_x], round(y1_values[index_x], 3),
                                               round(y2_values[index_x], 3), round(y3_values[index_x], 3)), end=" ")
            pof_population[pof_i, :] = population[index_x]
            pof_i = pof_i + 1
        pof_y1_values = self.fitness1(pof_population[:, 0], pof_population[:, 1])
        pof_y2_values = self.fitness2(pof_population[:, 0], pof_population[:, 1])
        pof_y3_values = self.fitness3(pof_population[:, 0], pof_population[:, 1])
        return pof_population, pof_y1_values, pof_y2_values, pof_y3_values


        ### 2.2 种群的量子形式初始化

    def initial_population_angle_np(self):
        '''种群初始化
        input:self(object):QGA类
        output:population_Angle(list):种群的量子角度列表
               population_Angle2(list):空的种群的量子角度列表，用于存储交叉后的量子角度列表
        '''
        init_popu_angle_np = np.random.random(size=(self.POP_SIZE, self.CHROMOSOME_LEN,self.X_COUNT))
        init_popu_angle_np = np.pi * 2 * init_popu_angle_np
        return init_popu_angle_np

    def population_qubit_np(self, population_angle):
        return np.sin(population_angle.copy())

    ### 2.3 计算适应度函数值
    def translation_qbit_to_bit_np(self, population_Q):
        '''将种群的量子列表转换为二进制列表
        input:self(object):QGA类
              population_Q(list):种群的量子列表
        output:population_Binary:种群的二进制列表
        '''
        initial_judge = np.random.random(size=( self.POP_SIZE, self.CHROMOSOME_LEN,self.X_COUNT))
        population_bit_np=np.array(np.square(population_Q)>initial_judge,np.int)  #几率幅的平方
        # print(population_bit_np.shape)
        return population_bit_np

    def trans_bit_to_x_value(self, population_bit_np):
        ##1.染色体的二进制表现形式转换为十进制并设置在[min_value,max_value]之间
        X = np.full((self.POP_SIZE,self.X_COUNT), 0.0)  ##存储所有参数的可能取值
        for i in range(len(population_bit_np)):
            tmp1 = []  ##存储一个参数的可能取值
            for j in range(self.X_COUNT):
                ##计算二进制对应的十进制数值
                # int("110100010",2) = 418
                # for m in range(len(population_bit[i][j])):
                #     total += population_bit[i][j][m] * math.pow(2, m)
                total = int("".join('%s' % id for id in list(population_bit_np[i,:,j])), 2)
                value = (total * (self.MAX_X - self.MIN_X)) / math.pow(2, self.CHROMOSOME_LEN) + self.MIN_X  ##将十进制数值坐落在[min_value,max_value]之间
                tmp1.append(value)
            X[i] = tmp1
        return X

    def fitness_non_distance(self, population_double_X_np):
        # fitness评价
        y1_values_double_np = self.fitness1(population_double_X_np[:, 0], population_double_X_np[:, 1])
        y2_values_double_np = self.fitness2(population_double_X_np[:, 0], population_double_X_np[:, 1])
        y3_values_double_np = self.fitness3(population_double_X_np[:, 0], population_double_X_np[:, 1])
        # 非支配解排序
        non_do_sorted_double_fronts_ls = self.fast_non_dominated_sort(y1_values_double_np, y2_values_double_np,
                                                                      y3_values_double_np)
        # 拥挤距离计算
        c_distance_double_ls = []
        for i in range(0, len(non_do_sorted_double_fronts_ls)):
            distance_front_ls = self.crowding_distance(y1_values_double_np, y2_values_double_np, y3_values_double_np,
                                                       non_do_sorted_double_fronts_ls[i])
            c_distance_double_ls.append(distance_front_ls)
        return c_distance_double_ls, non_do_sorted_double_fronts_ls,non_do_sorted_double_fronts_ls[0]

    def generation_first(self, population_angle_np):
        """
        #生成第一代,采用交叉和变异的方式
        :param population_angle_np:
        :return:
        """
        # 将角度转为量子位，一个X（x1,x2）解对应2*20个量子位，实际上只是用到了1*20个的量子位，sin(a)那部分
        population_qubit_np = self.population_qubit_np(population_angle_np)
        ## 将量子位转换为二进制形式，量子bit决定了编码的多样性，这是非常关键的。
        population_bit_np = self.translation_qbit_to_bit_np(population_qubit_np)
        population_X_np = self.trans_bit_to_x_value(population_bit_np)
        # 生成两倍的后代 Generating offsprings
        # 将原有的x_solution扩充为x_solution_double
        population_double_X_np = np.vstack((population_X_np, population_X_np))
        population_double_angle_np = np.vstack((population_angle_np, population_angle_np))
        # 采用交叉、旋转、变异来更新population_double_X_np，population_double_angle_np
        for i in range(POP_SIZE, POP_SIZE * 2):
            a1 = random.randint(0, POP_SIZE - 1)
            b1 = random.randint(0, POP_SIZE - 1)
            # 通过crossover和mutation的方式生成新的soultion
            population_double_X_np[i, :], population_double_angle_np[i, :, :] = self.crossover_and_mutation(
                population_angle_np[a1, :, :], population_angle_np[b1, :, :], 0)  ###改改改
        return population_double_X_np, population_double_angle_np

    def generation_second(self, gen_no, population_angle_np,first_POF):
        """
        # 生成第一代之后的各代,采用旋转和变异的方式
        :param gen_no:
        :param population_angle_np:
        :param first_POF:
        :return:
        """
        # 将角度转为量子位，一个X（x1,x2）解对应2*20个量子位，实际上只是用到了1*20个的量子位，sin(a)那部分
        population_qubit_np = self.population_qubit_np(population_angle_np)
        ## 将量子位转换为二进制形式，量子bit决定了编码的多样性，这是非常关键的。
        population_bit_np = self.translation_qbit_to_bit_np(population_qubit_np)
        population_X_np = self.trans_bit_to_x_value(population_bit_np)
        # 生成两倍的后代 Generating offsprings
        # 将原有的x_solution扩充为x_solution_double
        population_double_X_np = np.vstack((population_X_np, population_X_np))
        population_double_angle_np = np.vstack((population_angle_np, population_angle_np))
        # 采用交叉、旋转、变异来更新population_double_X_np，population_double_angle_np
        for i in range(POP_SIZE, POP_SIZE * 2):
            a1 = random.randint(0, POP_SIZE - 1)
            if len(first_POF)>0:
                b1 = random.randint(0, len(first_POF) )
            else:
                b1 = random.randint(0, POP_SIZE - 1)
            # 通过rotate和mutation的方式生成新的soultion
            population_double_X_np[i, :], population_double_angle_np[i, :, :] = self.rotate_and_mutation(population_angle_np[a1, :, :],population_angle_np[b1, :, :],  gen_no)  ###改改改
        return population_double_X_np, population_double_angle_np

    def rotate_and_mutation(self, chromosome_angle_a, chromosome_angle_target, gen_no):
        """
        :param chromosome_angle_a:
        :param chromosome_angle_target:
        :param gen_no:  当前进化的代数，然后用于改变交叉概率和变异概率
        :return: 交叉，Function to carry out the crossover
        """
        #开展量子旋转门
        chromosome_angle_a_new=self.rotate(chromosome_angle_a, chromosome_angle_target)
        #变异
        mutation_prob = random.random()
        if mutation_prob<self.MUTATION_PROB_THRESHOLD*(1-gen_no/MAX_GEN):    #改进为变长概率
            chromosome_angle_a_new = self.mutation(chromosome_angle_a_new)  #这里有优化的空间
            #到这里了
        chromosome_qubit_a=np.sin(chromosome_angle_a_new)
        initial_judge = np.random.random(size=(self.CHROMOSOME_LEN, self.X_COUNT))
        chromosome_bit_a_np = np.array(chromosome_qubit_a > initial_judge, np.int)
        ##计算二进制对应的十进制数值
        x_a=[]
        for j in range(self.X_COUNT):
            total_a = int("".join('%s' % id for id in list(chromosome_bit_a_np[:,j])), 2)
            x_1_a = (total_a * (self.MAX_X - self.MIN_X)) / math.pow(2,self.CHROMOSOME_LEN) + self.MIN_X
            x_a.append(x_1_a)
        return x_a,chromosome_angle_a_new


    def transfer_chromosome_to_x(self,chromosome):
        """
        将角度染色体转变为x
        :param chromosome:
        :return: x,qbit_sin,qbit_cos,bit_np
        """
        qbit_sin=np.sin(chromosome)
        qbit_cos=np.cos(chromosome)
        initial_judge = np.random.random(size=(self.CHROMOSOME_LEN, self.X_COUNT))
        bit_np = np.array(np.square(qbit_sin) > initial_judge, np.int)  # 几率幅的平方
        x = []  ##存储一个参数的可能取值
        for j in range(self.X_COUNT):
            ##计算二进制对应的十进制数值
            total = int("".join('%s' % id for id in list(bit_np[:, j])), 2)
            ##将十进制数值坐落在[min_value,max_value]之间
            value = (total * (self.MAX_X - self.MIN_X)) / math.pow(2,self.CHROMOSOME_LEN) + self.MIN_X
            x.append(value)
        return x,qbit_sin,qbit_cos,bit_np

    def rotate(self,chromosome_angle_a, chromosome_angle_target):
        """
        量子旋转
        :param chromosome_angle_a:
        :param chromosome_angle_target:
        :return: 得到旋转后的结果
        """
        ##2.初始化每一个量子位的旋转角度
        rotation_angle_np = np.full((self.CHROMOSOME_LEN,self.X_COUNT),fill_value=0.0)
        x_a,qbit_sin_a,qbit_cos_a,bit_a=self.transfer_chromosome_to_x(chromosome_angle_a)
        x_target,qbit_sin_target,qbit_cos_target,bit_target=self.transfer_chromosome_to_x(chromosome_angle_target)
        #目标函数评价
        y1_a=self.fitness1(x_a[0],x_a[1])
        y2_a=self.fitness2(x_a[0],x_a[1])
        y3_a=self.fitness3(x_a[0],x_a[1])
        y1_target = self.fitness1(x_target[0], x_target[1])
        y2_target = self.fitness2(x_target[0], x_target[1])
        y3_target = self.fitness3(x_target[0], x_target[1])
        #支配关系判别
        # 这其中的是目标函数
        # y1求最小，y2求最大，y3求最大
        if (y1_target < y1_a and y2_target > y2_a and y3_target > y3_a):
            # 个体target的支配a计算
            ##3. 求每个量子位的旋转角度
            for m in range(self.CHROMOSOME_LEN):
                for n in range(self.X_COUNT):
                    s1 = 0
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s1 = -1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s1 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s1 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s1 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s1 = -1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s1 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s1 = -1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s1 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s1 = -1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s1 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s1 = -1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s1 = 1
                    rotation_angle_np[m,n] = self.ANGLE_DATE * s1
        else:
            for m in range(self.CHROMOSOME_LEN):
                for n in range(self.X_COUNT):
                    s2 = 0
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s2 = -1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s2 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s2 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s2 = -1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s2 = 1
                    if bit_a[m,n] == 0 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s2 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s2 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s2 = -1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 0 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s2 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] > 0:
                        s2 = 1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] < 0:
                        s2 = -1
                    if bit_a[m,n] == 1 and bit_target[m,n] == 1 and qbit_sin_a[m,n] * qbit_cos_a[m,n] == 0:
                        s2 = 1
                    rotation_angle_np[m,n] = self.ANGLE_DATE * s2
        ###4. 根据每个量子位的旋转角度，生成种群新的量子角度列表
        population_angle_rotate_np = chromosome_angle_a + rotation_angle_np
        ##5.变异后的适应度值求解
        return population_angle_rotate_np


    def new_generation_from_nondominated_fronts(self, c_distance_double_ls, non_do_sorted_double_fronts_ls):
        """
        生成新的一代，返回是用于筛选的index列表
        :param c_distance_double_ls:
        :param non_do_sorted_double_fronts_ls:
        :return: 新种群的index
        """
        index_new_popu_ls = []
        for i in range(0, len(non_do_sorted_double_fronts_ls)):
            non_dominated_sorted_solution2_1 = [
                self.get_index_of(non_do_sorted_double_fronts_ls[i][j], non_do_sorted_double_fronts_ls[i]) for j in
                range(0, len(non_do_sorted_double_fronts_ls[i]))]
            front22_ls = self.sort_by_values_max(non_dominated_sorted_solution2_1, c_distance_double_ls[i][:])
            front_ls = [non_do_sorted_double_fronts_ls[i][front22_ls[j]] for j in
                        range(0, len(non_do_sorted_double_fronts_ls[i]))]
            front_ls.reverse()
            for index in front_ls:
                index_new_popu_ls.append(index)
                if (len(index_new_popu_ls) == POP_SIZE):
                    break
            if (len(index_new_popu_ls) == POP_SIZE):
                break
        return index_new_popu_ls


    def execute_quantum_nsga2(self):
        """
        :return: NSGA2的主函数
        """
        ## 种群初始化
        population_angle_np = self.initial_population_angle_np()
        pof=[]
        gen_no = 0
        # 大的循环
        while (gen_no < MAX_GEN):
            print(gen_no)
            if gen_no==0:
                #生成第一代,采用交叉和变异的方式
                population_double_X_np, population_double_angle_np = self.generation_first( population_angle_np)
            else:
                # 生成第一代之后的各代,采用旋转和变异的方式
                # population_double_X_np, population_double_angle_np = self.generation_second(gen_no, population_angle_np,first_POF=pof)
                # 无旋转的方式
                population_double_X_np, population_double_angle_np = self.generation_first(population_angle_np)
            c_distance_double_ls, non_do_sorted_double_fronts_ls,pof = self.fitness_non_distance(population_double_X_np)
            # 生成新的一代，返回是用于筛选的index列表
            index_new_popu_ls = self.new_generation_from_nondominated_fronts(c_distance_double_ls,non_do_sorted_double_fronts_ls)
            # 从index列表中筛选出angle和X
            population_angle_np=np.array([population_double_angle_np[index] for index in index_new_popu_ls])
            population_X_np=np.array([population_double_X_np[index] for index in index_new_popu_ls])
            gen_no = gen_no + 1
        # 从最终的结果population中提取出结果
        pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np = self.get_result(population_X_np)
        #返回结果
        return pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np

####################################################END:Quantum_NSGA2类####################################################


if __name__=="__main__":
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 30
    X_COUNT = 2
    CROSSOVER_PROB__THRESHOLD = 0.5
    MUTATION_PROB__THRESHOLD = 0.5
    # Initialization
    MIN_X = 0
    MAX_X = 10
    DELATE = 2e-7
    ANGLE_DETA = 0.05 * np.pi
    DISTANCE_INFINTE = 44444444444444
    CHROMOSOME_LEN=16
    BEGIN_C_M=8
    nsga2_obj=Quantum_NSGA2(POP_SIZE,
                            MAX_GEN,
                            X_COUNT,
                            CROSSOVER_PROB__THRESHOLD,
                            MUTATION_PROB__THRESHOLD,
                            MIN_X,
                            MAX_X,
                            DELATE,
                            DISTANCE_INFINTE,
                            evaluation.y1,
                            evaluation.y2,
                            evaluation.y3,
                            CHROMOSOME_LEN,
                            BEGIN_C_M,
                            ANGLE_DETA)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np=nsga2_obj.execute_quantum_nsga2()  #执行
    evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np)



