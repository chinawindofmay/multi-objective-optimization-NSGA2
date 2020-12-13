
import math
import random
import matplotlib.pyplot as plt
import numpy as np

'''
参考链接：https://blog.csdn.net/quinn1994/article/details/80679528?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
疑惑点：拥挤度的计算sort_by_values()和distance[k]，代码的逻辑梳理
        交叉运算里不就是进行的变异吗？变异的概率是1？

'''

# 定义函数1
def function1(x):
    if x<1 or x==1:
        value = -x
    elif x>1 and (x<3 or x==3):
        value = x-2
    elif x>3 and (x<4 or x==4):
        value = 4-x
    elif x>4:
        value = x-4
    return value


# 定义函数2
def function2(x):
    value = (x - 5) ** 2
    return value


# Function to find index of list
# 查找列表指定元素的索引
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
# 函数根据指定的值列表排序
#list1是front，values是目标函数的值
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):# 当结果长度不等于初始长度时，继续循环
        #index_of(min(values), values)最小值的索引
        if index_of(min(values), values) in list1:# 标定值中最小值在目标列表中时
            sorted_list.append(index_of(min(values), values)) #将标定值的最小值的索引追加到结果列表后面
        values[index_of(min(values), values)] = math.inf    #将标定值的最小值置为无穷小,即删除原来的最小值,移向下一个
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
# 函数执行NSGA-II的快速非支配排序,将所有的个体都分层
'''
郭军p21
1.np=0 sp=infinite
2.对所有个体进行非支配判断，若p支配q，则将q加入到sp中，并将q的层级提升一级。
  若q支配p，将p加入sq中，并将p的层级提升一级。
3.对种群当前分层序号k进行初始化，令k=1
4.找出种群中np=0的个体，将其从种群中移除，将其加入到分层集合fk中，该集合就是层级为0个体的集合。
5.判断fk是否为空，若不为空，将fk中所有的个体sp中对应的个体层级减去1，且k=k+1,跳到2;
  若为空，则表明得到了所有非支配集合，程序结束
'''
"""基于序列和拥挤距离,这里找到任意两个个体p,q"""


def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    # 种群中所有个体的sp进行初始化 这里的len(value1)=pop_size
    front = [[]]
    # 分层集合,二维列表中包含第n个层中,有那些个体
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]
    # 评级
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        # 寻找第p个个体和其他个体的支配关系
        # 将第p个个体的sp和np初始化
        for q in range(0, len(values1)):
            # step2:p > q 即如果p支配q,则
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                # 支配判定条件:当且仅当,对于任取i属于{1,2},都有fi(p)>fi(q),符合支配.或者当且仅当对于任意i属于{1,2},有fi(p)>=fi(q),且至少存在一个j使得fj(p)>f(q)  符合弱支配
                n[p] = n[p] + 1
            # 如果q支配p
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                # 则将np+1
                if q not in S[p]:
                    # 同时如果q不属于sp将其添加到sp中
                    S[p].append(q)

        if n[p] == 0:#所有目标函数都取到了最优值，并将其划分为等级0
            # 找出种群中np=0的个体
            rank[p] = 0
            # 将其从pt中移去
            if p not in front[0]:
                # 如果p不在第0层中
                # 将其追加到第0层中
                front[0].append(p)
    i = 0
    while (front[i] != []):
        # 如果分层集合为不为空，
        Q = []
        for p in front[i]:#遍历p的支配集合
            for q in S[p]:
                n[q] = n[q] - 1
                # 则将fk中所有给对应的个体np-1
                if (n[q] == 0):
                    # 如果nq==0
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        # 并且k+1
        front.append(Q)
    #末尾会追加一个空的列表
    del front[len(front) - 1]
    # 返回将所有个体分层后的结果
    return front

# Function to calculate crowding distance
# 计算拥挤距离的函数
'''
高媛p29
1.I[1]=I[l]=inf，I[i]=0 将边界的两个个体拥挤度设为无穷。
2.I=sort(I,m)，基于目标函数m对种群排序
3.I[i]=I[i]+(Im[i+1]-Im[i-1])/(fmax-fmin)
'''

#values1, values2目标函数值
#front是每一级结果
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]#初始化所有个体的拥挤距离
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])   # 基于目标函数1和目标函数2对已经划分好层级的种群排序
    distance[0] = 4444444444444444#第一个点设置为无穷大
    distance[len(front) - 1] = 4444444444444444#最后一个点设为无穷大
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    # 返回拥挤距离
    return distance


# 函数进行交叉
def crossover(a, b):
    r = np.random.random()
    if r > 0.5:
        return mutation((a + b) / 2)
    else:
        return mutation((a - b) / 2)


# 函数进行变异操作
def mutation(solution):
    mutation_prob = np.random.random()
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * np.random.random()
    return solution


pop_size = 100#种群的规模
max_gen = 100#最大迭代次数
# 迭代次数
# Initialization
min_x = -5#最小取值
max_x = 10#最大取值

#随机生成pop_size个最小值到最大值之间的数字
solution = [min_x + (max_x - min_x) * np.random.random() for i in np.arange(0, pop_size)]
# 随机生成变量
gen_no = 0
while (gen_no < max_gen):
    # 生成两个函数值列表，构成一个种群
    function1_values = [function1(solution[i]) for i in np.arange(0, pop_size)]#目标函数1的函数值
    function2_values = [function2(solution[i]) for i in np.arange(0, pop_size)]
    # 种群之间进行快速非支配性排序,得到非支配性排序集合
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])#得到的是solution的索引
    print("The best front for Generation number ", gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez], 3), end=" ")#第0代对应的x的值
    print("\n")
    # 计算非支配集合中每个个体的拥挤度
    crowding_distance_values = []
    for i in range(0, len(non_dominated_sorted_solution)):
        #non_dominated_sorted_solution[i][:]遍历快速非支配排序结果的每一级
        crowding_distance_values.append(
            crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]

    # 生成了子代
    while (len(solution2) != 2 * pop_size):
        a1 = np.random.randint(0, pop_size - 1)
        b1 = np.random.randint(0, pop_size - 1)
        # 选择
        solution2.append(crossover(solution[a1], solution[b1]))#父代+遗传运算后的子代
        # 随机选择，将种群中的个体进行交配，得到子代种群2*pop_size

    function1_values2 = [function1(solution2[i]) for i in range(0, 2 * pop_size)]
    function2_values2 = [function2(solution2[i]) for i in range(0, 2 * pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
    # 将两个目标函数得到的两个种群值value,再进行排序得到2*pop_size解
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(
            crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
    # 计算子代的个体间的距离值
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [
            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
            range(0, len(non_dominated_sorted_solution2[i]))]
        # 排序
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                 range(0, len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if (len(new_solution) == pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

# Lets plot the final front now
function1 = [i for i in function1_values]
function2 = [j for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()