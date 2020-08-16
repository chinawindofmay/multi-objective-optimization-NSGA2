# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np


#Main program starts here
POP_SIZE = 20
MAX_GEN = 1000

CROSSOVER_PROB__THRESHOLD =0.5
MUTATION_PROB__THRESHOLD=0.6

#Initialization
MIN_X=-55
MAX_X=55

#First function to optimize
#希望求最大值
def y1(x):
    value = -x**2
    return value

#Second function to optimize
#希望求最大值
def y2(x):
    value = (x-10)**3
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
def fast_non_dominated_sort(y1_values, y2_values):
    S=[[] for i in range(0, len(y1_values))]
    fronts = [[]]
    n=[0 for i in range(0, len(y1_values))]
    rank = [0 for i in range(0, len(y1_values))]

    for p in range(0, len(y1_values)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(y1_values)):
            # 这其中中的是目标函数，发现都是求最大值
            if (y1_values[p] > y1_values[q] and y2_values[p] > y2_values[q]) or (y1_values[p] >= y1_values[q] and y2_values[p] > y2_values[q]) or (y1_values[p] > y1_values[q] and y2_values[p] >= y2_values[q]):
                if q not in S[p]:
                    # 个体p的支配集合Sp计算
                    S[p].append(q)
            elif (y1_values[q] > y1_values[p] and y2_values[q] > y2_values[p]) or (y1_values[q] >= y1_values[p] and y2_values[q] > y2_values[p]) or (y1_values[q] > y1_values[p] and y2_values[q] >= y2_values[p]):
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


#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort_min(y1_values, y2_values):
    S=[[] for i in range(0, len(y1_values))]
    fronts = [[]]
    n=[0 for i in range(0, len(y1_values))]
    rank = [0 for i in range(0, len(y1_values))]

    for p in range(0, len(y1_values)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(y1_values)):
            # 这其中中的是目标函数，发现都是求最大值
            if (y1_values[p] < y1_values[q] and y2_values[p] < y2_values[q]) or (y1_values[p] <= y1_values[q] and y2_values[p] < y2_values[q]) or (y1_values[p] < y1_values[q] and y2_values[p] <= y2_values[q]):
                if q not in S[p]:
                    # 个体p的支配集合Sp计算
                    S[p].append(q)
            elif (y1_values[p] > y1_values[q] and y2_values[p] > y2_values[q]) or (y1_values[p] >= y1_values[q] and y2_values[p] > y2_values[q]) or (y1_values[p] > y1_values[q] and y2_values[p] >= y2_values[q]):
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
def crowding_distance(y1_values, y2_values, front):
    distance = [0 for i in range(0,len(front))]
    #根据y1的值做一次排序
    sorted1 = sort_by_values(front, y1_values[:])
    #根据y2的值做一次排序
    sorted2 = sort_by_values(front, y2_values[:])
    #第一个个体和最后一个个体，定义为无限远
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    #计算中间个体的距离
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted1[k + 1]] - y1_values[sorted1[k - 1]]) / (max(y1_values) - min(y1_values))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y2_values[sorted2[k + 1]] - y2_values[sorted2[k - 1]]) / (max(y2_values) - min(y2_values))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    crossover_prob=random.random()
    if crossover_prob>CROSSOVER_PROB__THRESHOLD:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(x):
    mutation_prob = random.random()
    if mutation_prob <MUTATION_PROB__THRESHOLD:
        x = MIN_X + (MAX_X - MIN_X) * random.random()
    return x

#三维制图表达
def draw_3d_plot(x_solution,y1_values,y2_values):
    # fig = plt.figure()  # 相当于一个画板
    fig = plt.figure(figsize=(16, 16))
    from mpl_toolkits.mplot3d import Axes3D
    ax11 = Axes3D(fig)
    # set figure information
    ax11.set_title("NSGA2 Value")
    ax11.set_xlabel("x")
    ax11.set_ylabel("y1")
    ax11.set_zlabel("y2")
    ax11.scatter(x_solution, y1_values, y2_values, c='r')
    plt.show()

#二维制图表达
def draw_2d_combined_plot(x_solution,y1_values,y2_values):
    fig = plt.figure(figsize=(4, 12))
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)
    ax11.set_xlabel('x', fontsize=15)
    ax11.set_ylabel('y1', fontsize=15)
    ax11.scatter(x_solution, y1_values)
    ax12.set_xlabel('x', fontsize=15)
    ax12.set_ylabel('y2', fontsize=15)
    ax12.scatter(x_solution, y2_values)
    ax13.set_xlabel('y1', fontsize=15)
    ax13.set_ylabel('y2', fontsize=15)
    ax13.scatter(y1_values, y2_values)
    plt.show()

#NSGA2的主函数
def execute_nsga2():
    x_solution_popu = [MIN_X + (MAX_X - MIN_X) * random.random() for i in range(0, POP_SIZE)]
    gen_no = 0
    #大的循环
    while (gen_no < MAX_GEN):
        # 输出当前代的最优非支配曲面上的结果
        y1_values = [y1(x_solution_popu[i]) for i in range(0, POP_SIZE)]
        y2_values = [y2(x_solution_popu[i]) for i in range(0, POP_SIZE)]
        non_dominated_sorted_fronts = fast_non_dominated_sort(y1_values[:], y2_values[:])
        print("The best front for Generation number ", gen_no, " is")
        # 获取POF，最优非支配曲面(Pareto - optimal front)
        for index_x in non_dominated_sorted_fronts[0]:
            print("x={0},y1={1},y2={2}".format(round(x_solution_popu[index_x], 3), round(y1_values[index_x], 3),
                                               round(y2_values[index_x], 3)), end=" ")
            print("\n")

        # 生成两倍的后代 Generating offsprings
        # 将原有的x_solution扩充为x_solution_double
        x_solution_popu_double = x_solution_popu[:]
        while (len(x_solution_popu_double) != 2 * POP_SIZE):
            a1 = random.randint(0, POP_SIZE - 1)
            b1 = random.randint(0, POP_SIZE - 1)
            # 通过crossover和mutation的方式生成新的个体
            x_solution_popu_double.append(crossover(x_solution_popu[a1], x_solution_popu[b1]))
        # 评价值
        y1_values2 = [y1(x_solution_popu_double[i]) for i in range(0, 2 * POP_SIZE)]
        y2_values2 = [y2(x_solution_popu_double[i]) for i in range(0, 2 * POP_SIZE)]
        # 测试代码
        Y1=[9,7,5,4,3,2,1,10,8,7,5,4,3,10,9,8,7,10,9,8]
        Y2=[1,2,4,5,6,7,9,1,5,6,7,8,9,5,6,7,9,6,7,9]
        # 排序
        non_do_sorted_double_fronts = fast_non_dominated_sort(y1_values2[:], y2_values2[:])
        # 计算拥挤度
        c_distance_double = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            c_distance_double.append(crowding_distance(y1_values2[:], y2_values2[:], non_do_sorted_double_fronts[i][:]))

        # 生成新的一代
        index_list_new_solution_popu = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            non_dominated_sorted_solution2_1 = [
                get_index_of(non_do_sorted_double_fronts[i][j], non_do_sorted_double_fronts[i]) for j in
                range(0, len(non_do_sorted_double_fronts[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], c_distance_double[i][:])
            front = [non_do_sorted_double_fronts[i][front22[j]] for j in range(0, len(non_do_sorted_double_fronts[i]))]
            front.reverse()
            for value in front:
                index_list_new_solution_popu.append(value)
                if (len(index_list_new_solution_popu) == POP_SIZE):
                    break
            if (len(index_list_new_solution_popu) == POP_SIZE):
                break
        x_solution_popu = [x_solution_popu_double[i] for i in index_list_new_solution_popu]
        gen_no = gen_no + 1
    # 将最后一代的fitness结果打印出来
    #支配解的分布查看
    draw_2d_combined_plot(x_solution_popu, y1_values, y2_values)
    # draw_3d_plot(x_solution_popu, y1_values, y2_values)

#非支配求解是对的
def test_ndset():
    Y1 = [9, 7, 5, 4, 3, 2, 1, 10, 8, 7, 5, 4, 3, 10, 9, 8, 7, 10, 9, 8]
    Y2 = [1, 2, 4, 5, 6, 7, 9, 1, 5, 6, 7, 8, 9, 5, 6, 7, 9, 6, 7, 9]
    # 排序
    non_do_sorted_double_fronts = fast_non_dominated_sort_min(Y1, Y2)
    for item in non_do_sorted_double_fronts:
        print(item)
        print("\n")
    # 正确的答案是：
    #     [0, 1, 2, 3, 4, 5, 6]
    #     [7, 8, 9, 10, 11, 12]
    #     [13, 14, 15, 16]
    #     [17, 18, 19]

if __name__=="__main__":
    # execute_nsga2()
    test_ndset()

