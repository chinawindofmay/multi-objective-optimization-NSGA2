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

####################################################BEGIN:NSGA2类####################################################
class NSGA2:
    def __init__(self,POP_SIZE,MAX_GEN,X_COUNT,CROSSOVER_PROB__THRESHOLD,MUTATION_PROB__THRESHOLD,MIN_X,MAX_X,DELATE,DISTANCE_INFINTE,fitness1,fitness2,fitness3):
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
        self.fitness3=fitness3

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

    def crossover(self,solution_a, solution_b):
        """
        :param solution_a:
        :param solution_b:
        :return: 交叉，Function to carry out the crossover
        """
        crossover_prob = random.random()
        if crossover_prob < CROSSOVER_PROB__THRESHOLD:
            return self.limitation(
                [self.mutation((solution_a[0] + solution_b[0]) / 2), self.mutation((solution_a[1] + solution_b[1]) / 2)])
        else:
            return solution_a

    def limitation(self,solution):
        """
        :param solution:
        :return:限制
        """
        new_solution = []
        for item in solution:
            if item < MIN_X or item > MAX_X:
                item = MIN_X + (MAX_X - MIN_X) * random.random()
            new_solution.append(item)
        return new_solution

    def mutation(self,x):
        """
        :param x:
        :return: 变异，Function to carry out the mutation operator
        """
        mutation_prob = random.random()
        if mutation_prob < MUTATION_PROB__THRESHOLD:
            x = MIN_X + (MAX_X - MIN_X) * random.random()
        return x

    def get_result(self, population):
        """
        :param population:
        :return: 将进化出的结果，进一步提取最优的进行制图表达
        """
        # 输出当前代的最优非支配曲面上的结果
        y1_values = self.fitness1(population[:, 0], population[:, 1])
        y2_values = self.fitness2(population[:, 0], population[:, 1])
        y3_values = self.fitness3(population[:, 0], population[:, 1])
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

    def execute_nsga2(self):
        """
        :return: NSGA2的主函数
        """
        population_np = np.random.uniform(MIN_X, MAX_X, (POP_SIZE, X_COUNT))  # 存放x1,x2
        gen_no = 0
        # 大的循环
        while (gen_no < MAX_GEN):
            print(gen_no)
            # 生成两倍的后代 Generating offsprings
            # 将原有的x_solution扩充为x_solution_double
            population_double_np = np.vstack((population_np, population_np))
            for i in range(POP_SIZE, POP_SIZE * 2):
                a1 = random.randint(0, POP_SIZE - 1)
                b1 = random.randint(0, POP_SIZE - 1)
                # 通过crossover和mutation的方式生成新的soultion
                population_double_np[i, :] = self.crossover(population_np[a1, :], population_np[b1, :])
            # 评价值
            y1_values_double_np = self.fitness1(population_double_np[:, 0], population_double_np[:, 1])
            y2_values_double_np = self.fitness2(population_double_np[:, 0], population_double_np[:, 1])
            y3_values_double_np = self.fitness3(population_double_np[:, 0], population_double_np[:, 1])
            # 排序
            non_do_sorted_double_fronts_ls = self.fast_non_dominated_sort(y1_values_double_np, y2_values_double_np, y3_values_double_np)
            # 计算拥挤度
            c_distance_double_ls = []
            for i in range(0, len(non_do_sorted_double_fronts_ls)):
                distance_front_ls = self.crowding_distance(y1_values_double_np, y2_values_double_np, y3_values_double_np,non_do_sorted_double_fronts_ls[i])
                c_distance_double_ls.append(distance_front_ls)
            # 生成新的一代
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
            population_np = np.array([population_double_np[index] for index in index_new_popu_ls])
            gen_no = gen_no + 1
        # 从population中提取出结果
        pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np = self.get_result(population_np)
        return pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np

####################################################END:NSGA2类####################################################


####################################################BEGIN：工具函数################################################
#希望求最小值
#三维平面
def y1(x1,x2):
    value = 200-3*x1-7*x2
    return value

#希望求最大值
#抛物面
def y2(x1,x2):
    value = (x1-1)**2+(x2-2)**2
    return value

#希望求最大值
#选择曲面
def y3(x1,x2):
    value = (x1**2+x2**2)**0.5
    return value

#三维制图表达
def draw_3d_plot(population, y1_values,y2_values,y3_values):
    """
    :param population:
    :param y1_values:
    :param y2_values:
    :param y3_values:
    :return: 三维制图
    """
    fig = plt.figure(figsize=(16, 16))
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = Axes3D(fig)
    # set figure information
    ax3d.set_title("The comparison between formula result and NSGA2 result")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("y")
    x1_list=[ solution[0] for solution in population]
    x2_list=[ solution[1] for solution in population]
    ax3d.scatter(x1_list, x2_list, y1_values, c='r',marker="v",linewidths=4)
    ax3d.scatter(x1_list, x2_list, y2_values, c='g',marker="*",linewidths=4)
    ax3d.scatter(x1_list, x2_list, y3_values, c='b',marker=".",linewidths=4)
    #曲面y1的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    def f1(x1, x2):
        return (200-3*x1-7*x2)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f1(x1, x2), rstride=1, cstride=1, cmap=plt.cm.spring)

    # 曲面y2的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    def f2(x1, x2):
        return ((x1-1)**2+(x2-2)**2)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f2(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)

    # 曲面y3的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    def f3(x1, x2):
        return (x1**2+x2**2)**0.5
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f3(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    plt.show()

#三维制图表达
def draw_3d_plot_test3333():
    """
    :return: 测试代码
    """
    fig = plt.figure(figsize=(16, 16))
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = Axes3D(fig)
    # set figure information
    ax3d.set_title("The comparison between formula result and NSGA2 result")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("y")
    # # 曲面y3的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    def f3(x1, x2):
        return (x1**2+x2**2)**0.5
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f3(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    plt.show()

# #二维制图表达
# def draw_2d_plot(y1_values,y2_values):
#     fig = plt.figure(figsize=(12, 12))
#     ax11 = fig.add_subplot(111)
#     ax11.set_xlabel('y1', fontsize=15)
#     ax11.set_ylabel('y2', fontsize=15)
#     ax11.scatter(y1_values, y2_values)
#     plt.show()
####################################################END:工具函数#####################################


if __name__=="__main__":
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 200
    X_COUNT = 2
    CROSSOVER_PROB__THRESHOLD = 0.4
    MUTATION_PROB__THRESHOLD = 0.05
    # Initialization
    MIN_X = 0
    MAX_X = 10
    DELATE = 2e-7
    DISTANCE_INFINTE = 44444444444444
    nsga2_obj=NSGA2(POP_SIZE,MAX_GEN,X_COUNT,CROSSOVER_PROB__THRESHOLD,MUTATION_PROB__THRESHOLD,MIN_X,MAX_X,DELATE,DISTANCE_INFINTE,y1,y2,y3)
    pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np=nsga2_obj.execute_nsga2()
    draw_3d_plot(pof_population_np, pof_y1_values_np, pof_y2_values_np, pof_y3_values_np)
    # draw_3d_plot_test3333()



