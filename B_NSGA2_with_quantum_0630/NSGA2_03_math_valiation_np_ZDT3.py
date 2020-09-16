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
from B_NSGA2_with_quantum_0630 import NSGA2_05_evaluation as Evaluation


####################################################BEGIN:NSGA2类####################################################
class Traditional_NSGA2:
    def __init__(self,POP_SIZE,MAX_GEN,X_COUNT,CROSSOVER_PROB__THRESHOLD,MUTATION_PROB__THRESHOLD,MIN_X,MAX_X,DELATE,DISTANCE_INFINTE,fitness1,fitness2):
        self.POP_SIZE = POP_SIZE
        self.MAX_GEN = MAX_GEN
        self.X_COUNT = X_COUNT
        self.CROSSOVER_PROB_THRESHOLD = CROSSOVER_PROB__THRESHOLD
        self.MUTATION_PROB_THRESHOLD = MUTATION_PROB__THRESHOLD
        # Initialization
        self.MIN_X = MIN_X
        self.MAX_X = MAX_X
        self.DELATE = DELATE
        self.DISTANCE_INFINTE = DISTANCE_INFINTE
        self.fitness1=fitness1
        self.fitness2=fitness2

    def zdt4(self,MIN_X_I,MAX_X_I):
        self.MIN_X_I=MIN_X_I
        self.MAX_X_I=MAX_X_I

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
        return np.array(sorted_list)

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

    def fast_non_dominated_sort(self,y1_values, y2_values):
        """
        :param y1_values:
        :param y2_values:
        :return: 非支配解排序 Function to carry out NSGA-II's fast non dominated sort
        """
        # 存放p的非支配解
        nd_saver = [[] for i in range(0, len(y1_values))]
        nd_fronts = [[]]
        #存放p的被支配度
        d_degree = [0 for i in range(0, len(y1_values))]
        rank = [0 for i in range(0, len(y1_values))]
        # 求解非支配解
        for p_index in range(0, len(y1_values)):
            # nd_saver[p_index] = []
            # d_degree[p_index] = 0
            for q_index in range(0, len(y1_values)):
                if p_index==q_index:
                    continue
                # 这其中的是目标函数
                # y1求最小，y2求最小
                if (y1_values[p_index]-self.DELATE < y1_values[q_index] and y2_values[p_index]-self.DELATE < y2_values[q_index] ):
                    if q_index not in nd_saver[p_index]:
                        # 个体p的支配集合Sp计算
                        nd_saver[p_index].append(q_index)
                # 支配的定义为如果个体p中所有目标均不优于个体q中对应目标，则称个体p被个体q所支配。
                elif (y1_values[p_index] > y1_values[q_index]-self.DELATE and y2_values[p_index] > y2_values[q_index]-self.DELATE ):
                    # 被支配度Np计算
                    # Np越大，则说明p个体越差
                    d_degree[p_index] = d_degree[p_index] + 1
            if d_degree[p_index] == 0:
                rank[p_index] = 0
                if p_index not in nd_fronts[0]:
                    nd_fronts[0].append(p_index)
        i = 0
        while (nd_fronts[i] != []):
            Q = []
            for p_index in nd_fronts[i]:
                for q_index in nd_saver[p_index]:
                    d_degree[q_index] = d_degree[q_index] - 1
                    if (d_degree[q_index] == 0):
                        rank[q_index] = i + 1
                        if q_index not in Q:
                            Q.append(q_index)
            i = i + 1
            nd_fronts.append(Q)
        del nd_fronts[len(nd_fronts) - 1]
        return nd_fronts

    def crowding_distance(self, y1_values, y2_values, front_p):
        """
        :param y1_values:
        :param y2_values:
        :param front_p:
        :return: 拥挤距离计算，Function to calculate crowding distance
        """
        distance_ls = [0 for i in range(0, len(front_p))]
        # 根据y1的值做一次排序
        sorted1 = self.sort_by_values_min(front_p, y1_values.copy())
        # 根据y2的值做一次排序
        sorted2 = self.sort_by_values_min(front_p, y2_values.copy())
        index_index_1_min=np.where(front_p==sorted1[0])[0][0]
        index_index_1_max=np.where(front_p==sorted1[-1])[0][0]
        index_index_2_min=np.where(front_p==sorted2[0])[0][0]
        index_index_2_max=np.where(front_p==sorted2[-1])[0][0]
        # 第一个个体和最后一个个体，定义为无限远
        distance_ls[index_index_1_min] = self.DISTANCE_INFINTE
        distance_ls[index_index_1_max] = self.DISTANCE_INFINTE
        distance_ls[index_index_2_min] = self.DISTANCE_INFINTE
        distance_ls[index_index_2_max] = self.DISTANCE_INFINTE
        distance_y1 = np.max(y1_values) - np.min(y1_values) + self.DELATE
        distance_y2 = np.max(y2_values) - np.min(y2_values) + self.DELATE
        # 计算中间个体的距离
        for k in range(len(front_p)):
            if distance_ls[k]==0:
                index=front_p[k]
                index_index_3 = np.where(sorted1 == index)[0][0]
                index_index_4 = np.where(sorted2 == index)[0][0]
                distance_ls[k] = distance_ls[k] + (y1_values[sorted1[index_index_3 + 1]] - y1_values[sorted1[index_index_3 - 1]]) / distance_y1
                distance_ls[k] = distance_ls[k] + (y2_values[sorted2[index_index_4 + 1]] - y2_values[sorted2[index_index_4 - 1]]) / distance_y2
        # distance_ls[0] = self.DISTANCE_INFINTE
        # distance_ls[len(front_p) - 1] = self.DISTANCE_INFINTE
        # distance_y1 = np.max(y1_values) - np.min(y1_values) + self.DELATE
        # distance_y2 = np.max(y2_values) - np.min(y2_values) + self.DELATE
        # # 计算中间个体的距离
        # for k in range(1, len(front_p) - 1):
        #     distance_ls[k] = distance_ls[k] + (y1_values[sorted1[k + 1]] - y1_values[sorted1[k - 1]]) / distance_y1
        #     distance_ls[k] = distance_ls[k] + (y1_values[sorted2[k + 1]] - y1_values[sorted2[k - 1]]) / distance_y1
        #     distance_ls[k] = distance_ls[k] + (y2_values[sorted1[k + 1]] - y2_values[sorted1[k - 1]]) / distance_y2
        #     distance_ls[k] = distance_ls[k] + (y2_values[sorted2[k + 1]] - y2_values[sorted2[k - 1]]) / distance_y2
        return distance_ls

    def crossover(self,solution_a, solution_b,change_ratio):
        """
        :param solution_a:
        :param solution_b:
        :return: 交叉，Function to carry out the crossover
        """
        crossover_prob = random.random()
        if crossover_prob < self.CROSSOVER_PROB_THRESHOLD*change_ratio:
            return self.limitation(
                [(solution_a[j] + solution_b[j]) / 2 for j in range(X_COUNT)])
        else:
            return solution_a

    def limitation(self,solution):
        """
        :param solution:
        :return:限制
        """
        new_solution = []
        for item in solution:
            if item < self.MIN_X or item > self.MAX_X:
                item = self.MIN_X + (self.MAX_X - self.MIN_X) * random.random()
            new_solution.append(item)
        return new_solution

    def mutation(self,x,change_ratio):
        """
        :param x:
        :return: 变异，Function to carry out the mutation operator
        """
        mutation_prob = random.random()
        if mutation_prob < self.MUTATION_PROB_THRESHOLD*change_ratio:
            x = self.MIN_X + (self.MAX_X - self.MIN_X) * random.random()
        return x

    def get_result(self, population):
        """
        :param population:
        :return: 将进化出的结果，进一步提取最优的进行制图表达
        """
        # 输出当前代的最优非支配曲面上的结果
        y1_values = self.fitness1(population)
        y2_values = self.fitness2(population)
        nd_fronts = self.fast_non_dominated_sort(y1_values[:], y2_values[:])
        len_pof = len(nd_fronts[0])
        pof_population = np.random.uniform(self.MIN_X, self.MAX_X, (len_pof, self.X_COUNT))  # 存放x1,x2
        # 获取POF，最优非支配曲面(Pareto - optimal front)
        pof_i = 0
        for index_x in nd_fronts[0]:
            print("x={0},y1={1},y2={2}".format(population[index_x], round(y1_values[index_x], 3),
                                               round(y2_values[index_x], 3)), end=" ")
            pof_population[pof_i, :] = population[index_x]
            pof_i = pof_i + 1
        pof_y1_values = self.fitness1(pof_population)
        pof_y2_values = self.fitness2(pof_population)
        return pof_population, pof_y1_values, pof_y2_values

    def evaluation_generation_distance(self,y1_values_double_np, y2_values_double_np,pof):
        generation_distance=0
        for index_y in range(self.POP_SIZE):
            for index_pof in range(len(pof)):
                generation_distance=generation_distance+np.power((y1_values_double_np[index_y]-y1_values_double_np[pof[index_pof]]),2)+np.power((y2_values_double_np[index_y] - y2_values_double_np[pof[index_pof]]), 2)
        generation_distance=np.power(generation_distance,0.5)/self.POP_SIZE
        return generation_distance

    def evaluation_space_presentation(self,y1_values_double_np, y2_values_double_np):
        di_np_array=np.full(self.POP_SIZE,0.0)
        for index_i in range(self.POP_SIZE):
            Di=np.abs(y1_values_double_np[index_i]-y1_values_double_np)+np.abs(y2_values_double_np[index_i]-y2_values_double_np)
            Di_temp=Di[np.where(Di>0)]  #过滤掉0值
            di=np.min(Di_temp)
            di_np_array[index_i]=di
        space_presentation=np.power(np.var(di_np_array)/(self.POP_SIZE-1),0.5)
        return space_presentation

    def execute_nsga2(self):
        """
        :return: NSGA2的主函数
        """
        # 从一个均匀分布[low, high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        population_np = np.random.uniform(self.MIN_X, self.MAX_X, (self.POP_SIZE, self.X_COUNT))
        gen_no = 0
        #用于评价NSGA2的效果
        gd_array = np.full(self.MAX_GEN, 0.0)
        sp_array = np.full(self.MAX_GEN, 0.0)
        # 大的循环
        while (gen_no < self.MAX_GEN):
            # 生成两倍的后代 Generating offsprings
            # 将原有的x_solution扩充为x_solution_double
            population_double_np = np.vstack((population_np, population_np))
            # 生成新的offerspring
            for i in range(self.POP_SIZE, self.POP_SIZE * 2):
                a1 = random.randint(0, self.POP_SIZE - 1)
                b1 = random.randint(0, self.POP_SIZE - 1)
                # 通过crossover和mutation的方式生成新的soultion
                # -gen_no / self.MAX_GEN
                solution_i = self.crossover(population_np[a1, :], population_np[b1, :],1)
                population_double_np[i, :] = self.mutation(solution_i,1)
            # 评价值
            y1_values_double_np = self.fitness1(population_double_np)
            y2_values_double_np = self.fitness2(population_double_np)
            # 非支配解快速排序
            nd_fronts_double_ls = self.fast_non_dominated_sort(y1_values_double_np, y2_values_double_np)
            # 计算拥挤度
            cd_double_ls = []
            for i in range(0, len(nd_fronts_double_ls)):
                distance_front_ls = self.crowding_distance(y1_values_double_np, y2_values_double_np,nd_fronts_double_ls[i])
                cd_double_ls.append(distance_front_ls)
            #评价
            # generation distance 计算
            # gd = self.evaluation_generation_distance(y1_values_double_np, y2_values_double_np,
            #                                          nd_fronts_double_ls[0])
            # sp = self.evaluation_space_presentation(y1_values_double_np, y2_values_double_np)
            # gd_array[gen_no] = gd
            # sp_array[gen_no] = sp
            # 从曲面上旋转生成新的一代
            index_list_of_new_population = []
            for i in range(0, len(nd_fronts_double_ls)):
                front_index_index = [
                    self.get_index_of(nd_fronts_double_ls[i][j], nd_fronts_double_ls[i]) for j in
                    range(0, len(nd_fronts_double_ls[i]))]
                #按照距离最小的方式排序，排序的结果是index的列表
                front_sorted_index_index_ls = self.sort_by_values_min(front_index_index, cd_double_ls[i][:])
                front_sorted_index_ls = [nd_fronts_double_ls[i][front_sorted_index_index_ls[j]] for j in
                         range(0, len(nd_fronts_double_ls[i]))]
                front_sorted_index_ls.reverse()
                for index in front_sorted_index_ls:
                    index_list_of_new_population.append(index)
                    if (len(index_list_of_new_population) == self.POP_SIZE):
                        break
                if (len(index_list_of_new_population) == self.POP_SIZE):
                    break
            population_np = np.array([population_double_np[index] for index in index_list_of_new_population])
            print("第{0}代，POF个数为{1}个".format(gen_no, len(nd_fronts_double_ls[0])))
            gen_no = gen_no + 1
        # 从population中提取出结果
        pof_population_np, pof_y1_values_np, pof_y2_values_np = self.get_result(population_np)
        return pof_population_np, pof_y1_values_np, pof_y2_values_np,gd_array,sp_array

####################################################END:NSGA2类####################################################



def ZDT3_test():
    global X_COUNT
    # Main program starts here
    POP_SIZE = 136
    MAX_GEN = 500
    X_COUNT = 30
    CROSSOVER_PROB__THRESHOLD = 0.9
    MUTATION_PROB__THRESHOLD = 1/30
    # Initialization
    MIN_X = 0
    MAX_X = 1
    DELATE = 2e-7
    DISTANCE_INFINTE = 444444
    nsga2_obj = Traditional_NSGA2(POP_SIZE, MAX_GEN, X_COUNT, CROSSOVER_PROB__THRESHOLD, MUTATION_PROB__THRESHOLD,
                                  MIN_X, MAX_X, DELATE, DISTANCE_INFINTE, Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, gd_array, sp_array = nsga2_obj.execute_nsga2()
    # 将GD、SP结果展示出来
    # Evaluation.draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array)
    # 将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_2(pof_y1_values_np, pof_y2_values_np)
    # ZDT3是一个超平面，无法展示出来
    # Evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np,Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    # draw_3d_plot_test3333()


def ZDT4_test():
    global X_COUNT
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 100
    X_COUNT = 10
    CROSSOVER_PROB__THRESHOLD = 0.8
    MUTATION_PROB__THRESHOLD = 0.1
    # Initialization
    MIN_X_1 = 0
    MAX_X_1 = 1
    MIN_X_I =-5
    MAX_X_I = 5
    DELATE = 2e-7
    DISTANCE_INFINTE = 44
    nsga2_obj = Traditional_NSGA2(POP_SIZE, MAX_GEN, X_COUNT, CROSSOVER_PROB__THRESHOLD, MUTATION_PROB__THRESHOLD,
                                  MIN_X_1, MAX_X_1, DELATE, DISTANCE_INFINTE, Evaluation.ZDT4_f1, Evaluation.ZDT4_f2)
    nsga2_obj.zdt4(MIN_X_I,MAX_X_I)
    ##########还有一部分没有改#################
    pof_population_np, pof_y1_values_np, pof_y2_values_np, gd_array, sp_array = nsga2_obj.execute_nsga2()
    # 将GD、SP结果展示出来
    # Evaluation.draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array)
    # 将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_2(pof_y1_values_np, pof_y2_values_np)
    # ZDT3是一个超平面，无法展示出来
    # Evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np,Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    # draw_3d_plot_test3333()


def ZDT6_test():
    global X_COUNT
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 1000
    X_COUNT = 10
    CROSSOVER_PROB__THRESHOLD = 0.8
    MUTATION_PROB__THRESHOLD = 0.1
    # Initialization
    MIN_X = 0
    MAX_X = 1
    DELATE = 2e-7
    DISTANCE_INFINTE = 444444
    nsga2_obj = Traditional_NSGA2(POP_SIZE, MAX_GEN, X_COUNT, CROSSOVER_PROB__THRESHOLD, MUTATION_PROB__THRESHOLD,
                                  MIN_X, MAX_X, DELATE, DISTANCE_INFINTE, Evaluation.ZDT6_f1, Evaluation.ZDT6_f2)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, gd_array, sp_array = nsga2_obj.execute_nsga2()
    # 将GD、SP结果展示出来
    # Evaluation.draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array)
    # 将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_2(pof_y1_values_np, pof_y2_values_np)
    # ZDT3是一个超平面，无法展示出来
    # Evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np,Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    # draw_3d_plot_test3333()


def DTLZ1_test():
    global X_COUNT
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 1000
    X_COUNT = 10
    CROSSOVER_PROB__THRESHOLD = 0.9
    MUTATION_PROB__THRESHOLD = 0.1
    # Initialization
    MIN_X = 0
    MAX_X = 1
    DELATE = 2e-7
    DISTANCE_INFINTE = 444444
    nsga2_obj = Traditional_NSGA2(POP_SIZE, MAX_GEN, X_COUNT, CROSSOVER_PROB__THRESHOLD, MUTATION_PROB__THRESHOLD,
                                  MIN_X, MAX_X, DELATE, DISTANCE_INFINTE, Evaluation.DTLZ1_f1, Evaluation.DTLZ1_f2, Evaluation.DTLZ1_f3)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, gd_array, sp_array = nsga2_obj.execute_nsga2()
    # 将GD、SP结果展示出来
    # Evaluation.draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array)
    # 将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_2(pof_y1_values_np, pof_y2_values_np)
    # ZDT3是一个超平面，无法展示出来
    # Evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np,Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    # draw_3d_plot_test3333()

def schaffer2_test():
    """
    需要把SP和GD给注释掉，可以跑出成功的结果。
    :return:
    """
    global X_COUNT
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 100
    X_COUNT = 1
    CROSSOVER_PROB__THRESHOLD = 0.8
    MUTATION_PROB__THRESHOLD = 0.1
    # Initialization
    MIN_X = -5
    MAX_X = 10
    DELATE = 2e-7
    DISTANCE_INFINTE = 444
    nsga2_obj = Traditional_NSGA2(POP_SIZE, MAX_GEN, X_COUNT, CROSSOVER_PROB__THRESHOLD, MUTATION_PROB__THRESHOLD,
                                  MIN_X, MAX_X, DELATE, DISTANCE_INFINTE, Evaluation.schaffer2_f1, Evaluation.schaffer2_f2)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, gd_array, sp_array = nsga2_obj.execute_nsga2()
    # 将GD、SP结果展示出来
    Evaluation.draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array)
    # 将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_2(pof_y1_values_np, pof_y2_values_np)
    # ZDT3是一个超平面，无法展示出来
    # Evaluation.draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np,Evaluation.ZDT3_f1, Evaluation.ZDT3_f2)
    # draw_3d_plot_test3333()





if __name__=="__main__":
    ZDT3_test()
    # ZDT6_test()
    # schaffer2_test()


