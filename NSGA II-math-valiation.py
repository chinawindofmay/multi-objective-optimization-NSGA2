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
from numpy import *


#Main program starts here
POP_SIZE = 100
MAX_GEN = 200

CROSSOVER_PROB__THRESHOLD =0.4
MUTATION_PROB__THRESHOLD=0.05

#Initialization
MIN_X=0
MAX_X=10

DELATE=2e-7
DISTANCE_INFINTE=444444444
#First function to optimize
#希望求最大值
#三维平面
def y1(x1,x2):
    value = 200-3*x1-7*x2
    return value

#Second function to optimize
#希望求最大值
#抛物面
def y2(x1,x2):
    value = (x1-1)**2+(x2-2)**2
    return value

#Second function to optimize
#希望求最大值
#选择曲面
def y3(x1,x2):
    value = (x1**2+x2**2)**0.5
    return value



#Function to find index of list
def get_index_of(a, list_obj):
    for i in range(0, len(list_obj)):
        if list_obj[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(front, y_values):
    sorted_list = []
    while(len(sorted_list)!=len(front)):
        if get_index_of(min(y_values), y_values) in front:
            sorted_list.append(get_index_of(min(y_values), y_values))
        y_values[get_index_of(min(y_values), y_values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(y1_values, y2_values,y3_values):
    S=[[] for i in range(0, len(y1_values))]
    fronts = [[]]
    n=[0 for i in range(0, len(y1_values))]
    rank = [0 for i in range(0, len(y1_values))]

    for p in range(0, len(y1_values)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(y1_values)):
            # 这其中中的是目标函数，发现都是求最大值
            if (y1_values[p] < y1_values[q] and y2_values[p] > y2_values[q] and y3_values[p] > y3_values[q])  :
                if q not in S[p]:
                    # 个体p的支配集合Sp计算
                    S[p].append(q)
            elif (y1_values[p] > y1_values[q] and y2_values[p] < y2_values[q] and y3_values[p] < y3_values[q]) :
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
    return fronts

#Function to calculate crowding distance
def crowding_distance(y1_values, y2_values,y3_values, front):
    distance = [0 for i in range(0,len(front))]
    #根据y1的值做一次排序
    sorted1 = sort_by_values(front, y1_values[:])
    #根据y2的值做一次排序
    sorted2 = sort_by_values(front, y2_values[:])
    sorted3 = sort_by_values(front, y3_values[:])
    #第一个个体和最后一个个体，定义为无限远
    distance[0] = DISTANCE_INFINTE
    distance[len(front) - 1] = DISTANCE_INFINTE
    #计算中间个体的距离
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted1[k + 1]] - y2_values[sorted1[k - 1]]) / (max(y1_values) - min(y1_values)+DELATE)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted2[k + 1]] - y2_values[sorted2[k - 1]]) / (max(y2_values) - min(y2_values)+DELATE)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted3[k + 1]] - y2_values[sorted3[k - 1]]) / (max(y2_values) - min(y2_values)+DELATE)
    return distance

#Function to carry out the crossover
def crossover(solution_a, solution_b):
    crossover_prob=random.random()
    if crossover_prob<CROSSOVER_PROB__THRESHOLD:
        return limitation([mutation((solution_a[0] + solution_b[0]) / 2),mutation((solution_a[1] + solution_b[1]) / 2)])
    else:
        return solution_a

def limitation(solution):
    new_solution=[]
    for item in solution:
        if item<MIN_X or item>MAX_X:
            item=MIN_X + (MAX_X - MIN_X) * random.random()
        new_solution.append(item)
    return new_solution

#Function to carry out the mutation operator
def mutation(x):
    mutation_prob = random.random()
    if mutation_prob <MUTATION_PROB__THRESHOLD:
        x = MIN_X + (MAX_X - MIN_X) * random.random()
    return x

#三维制图表达
def draw_3d_plot(population, y1_values,y2_values,y3_values):
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
    x1 = np.arange(MIN_X, MAX_X, 1)
    x2 = np.arange(MIN_X, MAX_X, 1)
    def f1(x1, x2):
        return (200-3*x1-7*x2)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f1(x1, x2), rstride=1, cstride=1, cmap=plt.cm.spring)

    # 曲面y2的
    x1 = np.arange(MIN_X, MAX_X, 1)
    x2 = np.arange(MIN_X, MAX_X, 1)
    def f2(x1, x2):
        return ((x1-1)**2+(x2-2)**2)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f2(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)

    # 曲面y3的
    x1 = np.arange(MIN_X, MAX_X, 1)
    x2 = np.arange(MIN_X, MAX_X, 1)
    def f3(x1, x2):
        return (x1**2+x2**2)**0.5
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f3(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    plt.show()

#三维制图表达
def draw_3d_plot_test3333():
    fig = plt.figure(figsize=(16, 16))
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = Axes3D(fig)
    # set figure information
    ax3d.set_title("The comparison between formula result and NSGA2 result")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("y")

    # # 曲面y3的
    x1 = np.arange(MIN_X, MAX_X, 1)
    x2 = np.arange(MIN_X, MAX_X, 1)
    def f3(x1, x2):
        return (x1**2+x2**2)**0.5
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f3(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    plt.show()


#二维制图表达
def draw_2d_plot(y1_values,y2_values):
    fig = plt.figure(figsize=(12, 12))
    ax11 = fig.add_subplot(111)
    ax11.set_xlabel('y1', fontsize=15)
    ax11.set_ylabel('y2', fontsize=15)
    ax11.scatter(y1_values, y2_values)
    plt.show()

#NSGA2的主函数
def execute_nsga2():
    population = [[MIN_X + (MAX_X - MIN_X) * random.random(),
                   MIN_X + (MAX_X - MIN_X) * random.random(),
                   MIN_X + (MAX_X - MIN_X) * random.random()] for i in range(0, POP_SIZE)]
    gen_no = 0

    #大的循环
    while (gen_no < MAX_GEN):
        print(gen_no)
        # 生成两倍的后代 Generating offsprings
        # 将原有的x_solution扩充为x_solution_double
        population_double = population[:]
        while (len(population_double) != 2 * POP_SIZE):
            a1 = random.randint(0, POP_SIZE - 1)
            b1 = random.randint(0, POP_SIZE - 1)
            # 通过crossover和mutation的方式生成新的soultion
            population_double.append(crossover(population[a1], population[b1]))
        # 评价值
        y1_values_double = [y1(population_double[i][0],population_double[i][1]) for i in range(0, 2 * POP_SIZE)]
        y2_values_double = [y2(population_double[i][0],population_double[i][1]) for i in range(0, 2 * POP_SIZE)]
        y3_values_double = [y3(population_double[i][0],population_double[i][1]) for i in range(0, 2 * POP_SIZE)]
        # 排序
        non_do_sorted_double_fronts = fast_non_dominated_sort(y1_values_double[:], y2_values_double[:],y3_values_double[:])
        # 计算拥挤度
        c_distance_double = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            c_distance_double.append(crowding_distance(y1_values_double[:], y2_values_double[:],y3_values_double[:], non_do_sorted_double_fronts[i][:]))

        # 生成新的一代
        index_list_new_popu = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            non_dominated_sorted_solution2_1 = [
                get_index_of(non_do_sorted_double_fronts[i][j], non_do_sorted_double_fronts[i]) for j in
                range(0, len(non_do_sorted_double_fronts[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], c_distance_double[i][:])
            front = [non_do_sorted_double_fronts[i][front22[j]] for j in range(0, len(non_do_sorted_double_fronts[i]))]
            front.reverse()
            for index in front:
                index_list_new_popu.append(index)
                if (len(index_list_new_popu) == POP_SIZE):
                    break
            if (len(index_list_new_popu) == POP_SIZE):
                break
        population = [population_double[i] for i in index_list_new_popu]
        gen_no = gen_no + 1


    # 输出当前代的最优非支配曲面上的结果
    y1_values = [y1(population[i][0], population[i][1]) for i in range(0, POP_SIZE)]
    y2_values = [y2(population[i][0], population[i][1]) for i in range(0, POP_SIZE)]
    y3_values = [y3(population[i][0], population[i][1]) for i in range(0, POP_SIZE)]
    non_dominated_sorted_fronts = fast_non_dominated_sort(y1_values[:], y2_values[:],y3_values[:])
    pof_population=[]
    # 获取POF，最优非支配曲面(Pareto - optimal front)
    for index_x in non_dominated_sorted_fronts[0]:
        print("x={0},y1={1},y2={2}".format(population[index_x], round(y1_values[index_x], 3),
                                           round(y2_values[index_x], 3),round(y3_values[index_x], 3)), end=" ")
        print("\n")
        pof_population.append(population[index_x])
    pof_y1_values = [y1(pof_population[i][0], pof_population[i][1]) for i in range(0, len(pof_population))]
    pof_y2_values = [y2(pof_population[i][0], pof_population[i][1]) for i in range(0, len(pof_population))]
    pof_y3_values = [y3(pof_population[i][0], pof_population[i][1]) for i in range(0, len(pof_population))]
    # 将最后一代的fitness结果打印出来
    # draw_2d_plot( pof_y1_values, pof_y2_values)
    draw_3d_plot(pof_population, pof_y1_values,pof_y2_values,pof_y3_values)


if __name__=="__main__":
    execute_nsga2()
    # draw_3d_plot_test3333()

