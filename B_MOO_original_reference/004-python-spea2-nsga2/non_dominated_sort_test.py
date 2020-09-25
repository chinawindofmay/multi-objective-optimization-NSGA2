
import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt


class Test_class():

    def __init__(self):
        # 测试代码
        Y1 = [9, 7, 5, 4, 3, 2, 1, 10, 8, 7, 5, 4, 3, 10, 9, 8, 7, 10, 9, 8]
        Y2 = [1, 2, 4, 5, 6, 7, 9, 1,  5, 6, 7, 8, 9,  5, 6, 7, 9,  6, 7, 9]
        self.objectives_fitness_zjh=np.array([Y1,Y2]).T
        self.objectives_fitness_8 = np.array([[0.6373723585181119, 9.089424920752537],
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

        self.objectives_fitness_20 = np.array([[0.01783689,1.02469787],
                                    [0.04471213,0.93037726],
                                    [0.0236877,0.96322404],
                                    [0.04938809,0.92641409],
                                    [0.07031985,0.83355839],
                                    [0.05199109,0.88398541],
                                    [0.05006115,0.91107188],
                                    [0.,1.18579768],
                                    [0.05358649,0.85849188],
                                    [0.01657041,1.0721905,],
                                    [0.10409921,0.84829613],
                                    [0.06643826,0.84941993],
                                    [0.05358649,0.89876683],
                                    [0.01740193,1.09732744],
                                    [0.04938809,0.94687415],
                                    [0.05199109,0.93536007],
                                    [0.02400905,1.11603965],
                                    [0.05358649,0.89107165],
                                    [0.,1.23996387],
                                    [0.01783689,1.11625582]])

    def test_fast_non_dominated_sort_1(self, objectives_fitness):
        #对华电小扎扎写的非支配排序的修改和调整
        # https: // blog.csdn.net / qq_36449201 / article / details / 81046586
        fronts = []  # Pareto前沿面
        fronts.append([])
        set_sp = []
        npp = np.zeros(2*10)
        rank = np.zeros(2*10)
        for i in range(2 * 10):
            temp = []
            for j in range(2 * 10):
                if j != i:
                    if (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1]) or \
                        (objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]) or \
                        (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]):
                        temp.append(j)
                    elif (objectives_fitness[i][0] >= objectives_fitness[j][0] and objectives_fitness[i][1] > objectives_fitness[j][1]) or \
                        (objectives_fitness[i][0] > objectives_fitness[j][0] and objectives_fitness[i][1] >= objectives_fitness[j][1]) or \
                        (objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1]):
                        npp[i] += 1  # j支配 i，np+1
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
        del fronts[len(fronts) - 1]
        self.output_fronts(fronts)

    def output_fronts(self, fronts):
        # test code
        sum_coun = 0
        for kk in range(len(fronts)):
            sum_coun += len(fronts[kk])
        print(sum_coun)
        print(fronts)

    def test_fast_non_dominated_sort_error(self, objectives_fitness):
        #对华电小扎扎写的非支配排序的修改和调整
        # https: // blog.csdn.net / qq_36449201 / article / details / 81046586
        fronts = []  # Pareto前沿面
        fronts.append([])
        set_sp = []
        npp = np.zeros(2*10)
        rank = np.zeros(2*10)
        for i in range(2 * 10):
            temp = []
            for j in range(2 * 10):
                if j != i:
                    if (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]) :
                        temp.append(j)
                    elif (objectives_fitness[i][0] > objectives_fitness[j][0] and objectives_fitness[i][1] > objectives_fitness[j][1]):
                        npp[i] += 1  # j支配 i，np+1
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
        del fronts[len(fronts) - 1]
        self.output_fronts(fronts)

    def output_fronts(self, fronts):
        # test code
        sum_coun = 0
        for kk in range(len(fronts)):
            sum_coun += len(fronts[kk])
        print(sum_coun)
        print(fronts)

    def test_fast_non_dominated_sort_2(self, objectives_fitness):
        #对Github Haris Ali Khan写的NSGA2的非支配排序函数的调整
        # coding:utf-8
        # Program Name: NSGA-II.py
        # Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
        # Author: Haris Ali Khan
        # Supervisor: Prof. Manoj Kumar Tiwari
        set_sp=[[] for i in range(0, np.shape(objectives_fitness)[0])]
        fronts = [[]]
        npp=[0 for i in range(0, np.shape(objectives_fitness)[0])]
        rank = [0 for i in range(0, np.shape(objectives_fitness)[0])]
        for i in range(0, np.shape(objectives_fitness)[0]):
            set_sp[i]=[]
            # npp[i]=0
            for j in range(0, np.shape(objectives_fitness)[0]):
                if i != j:
                    if (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1]) or \
                        (objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]) or \
                        (objectives_fitness[j][0] >= objectives_fitness[i][0] and objectives_fitness[j][1] >= objectives_fitness[i][1]):
                        # 个体p的支配集合Sp计算
                        set_sp[i].append(j)
                    elif (objectives_fitness[i][0] >= objectives_fitness[j][0] and objectives_fitness[i][1] > objectives_fitness[j][1]) or \
                        (objectives_fitness[i][0] > objectives_fitness[j][0] and objectives_fitness[i][1] >= objectives_fitness[j][1]) or \
                        (objectives_fitness[j][0] > objectives_fitness[i][0] and objectives_fitness[j][1] > objectives_fitness[i][1]):
                        # 被支配度Np计算
                        # Np越大，则说明i个体越差
                        npp[i] += 1  # j支配 i，np+1
            if npp[i]==0:
                rank[i] = 0
                if i not in fronts[0]:
                    fronts[0].append(i)
        i = 0
        while(fronts[i] != []):
            Q=[]
            for p in fronts[i]:
                for q in set_sp[p]:
                    npp[q] =npp[q] - 1
                    if( npp[q]==0):
                        rank[q]=i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            fronts.append(Q)
        del fronts[len(fronts)-1]
        self.output_fronts(fronts)


# NSGA2入口
if __name__ == '__main__':
    # NSGA = NSGA2(30, 100, 200)
    # NSGA.run()
    test_class=Test_class()

    # test zjh  普通
    print("测试1：普通")
    test_class.test_fast_non_dominated_sort_1(test_class.objectives_fitness_zjh)
    test_class.test_fast_non_dominated_sort_2(test_class.objectives_fitness_zjh)
    test_class.test_fast_non_dominated_sort_error(test_class.objectives_fitness_zjh)
    # test ZDT3 正确
    print("测试2：正确，重点看错误的函数test_fast_non_dominated_sort_error")
    test_class.test_fast_non_dominated_sort_1(test_class.objectives_fitness_20)
    test_class.test_fast_non_dominated_sort_2(test_class.objectives_fitness_20)
    test_class.test_fast_non_dominated_sort_error(test_class.objectives_fitness_20)
    # test ZDT6 错误
    print("测试3：ZDT6 错误，重点看错误的函数test_fast_non_dominated_sort_error")
    test_class.test_fast_non_dominated_sort_1(test_class.objectives_fitness_8)
    test_class.test_fast_non_dominated_sort_2(test_class.objectives_fitness_8)
    test_class.test_fast_non_dominated_sort_error(test_class.objectives_fitness_8)