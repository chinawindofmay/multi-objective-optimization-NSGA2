# -*- coding: utf-8 -*-
"""
程序功能：实现nsga2算法，测试函数为ZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
说明：遗传算子为二进制竞赛选择，模拟二进制交叉和多项式变异
作者：(晓风)
email: 18821709267@163.com 
最初建立时间：2018.10.10
最近修改时间：2018.10.10
参考论文：
A fast and Elitist Multiobjective Genetic Algorithm:NSGA-Ⅱ
Kalyanmoy Deb,Associate Member, IEEE, Amrit Pratap, Sameer Agarwal, and T.Meyarivan
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION
参考BLOG：
https://blog.csdn.net/qq_40434430/article/details/82876572
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
import copy
from mpl_toolkits.mplot3d import Axes3D
start=time.time()#开始计时

def fitness_fun_mate_infor(fitness_function_name):
    if fitness_function_name== 'ZDT1':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt1=np.loadtxt('ZDT1.txt')
        plt.scatter(zdt1[:,0],zdt1[:,1],marker='o',color='green',s=40)
        fun_txt=zdt1
    elif fitness_function_name== 'ZDT2':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt2=np.loadtxt('ZDT2.txt')
        plt.scatter(zdt2[:,0],zdt2[:,1],marker='o',color='green',s=40)
        fun_txt=zdt2
    elif fitness_function_name== 'ZDT3':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt3=np.loadtxt('ZDT3.txt')
        plt.scatter(zdt3[:,0],zdt3[:,1],marker='o',color='green',s=40)
        fun_txt=zdt3
    elif fitness_function_name== 'ZDT4':
        f_num=2;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array=np.array([0,-5,-5,-5,-5,-5,-5,-5,-5,-5],dtype=float)#决策变量的最小值
        X_max_array=np.array([1,5,5,5,5,5,5,5,5,5],dtype=float)#决策变量的最大值
        zdt4=np.loadtxt('ZDT4.txt')
        plt.scatter(zdt4[:,0],zdt4[:,1],marker='o',color='green',s=40)
        fun_txt=zdt4
    elif fitness_function_name== 'ZDT6':
        f_num=2;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt6=np.loadtxt('ZDT6.txt')
        plt.scatter(zdt6[:,0],zdt6[:,1],marker='o',color='green',s=40)
        fun_txt=zdt6
    elif fitness_function_name== 'DTLZ1':
        f_num=3;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        dtlz1=np.loadtxt('DTLZ1.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz1[:,0],dtlz1[:,1],dtlz1[:,2],c='g')
        fun_txt=dtlz1
    elif fitness_function_name== 'DTLZ2':
        f_num=3;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        dtlz2=np.loadtxt('DTLZ2.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz2[:,0],dtlz2[:,1],dtlz2[:,2],c='g')
        fun_txt=dtlz2
    plt.show()
    return f_num,x_num,X_min_array,X_max_array,fun_txt

class Population:
    def __init__(self,population_np,fitness_np,nnd_np,rank_np):
        self.population_np=population_np
        self.fitness_np=fitness_np
        self.nnd_np=nnd_np
        self.rank_np=rank_np

class NSGA2:

    def __init__(self,POP_SIZE, fitness_function_name, x_num, X_min_array, X_max_array, f_num,pc,pm,yital1,yital2):
        self.POP_SIZE=POP_SIZE
        self.FITNESS_NAME=fitness_function_name
        self.X_NUM=x_num
        self.X_MIN_ARRAY=X_min_array
        self.X_MAX_ARRAY=X_max_array
        self.F_NUM=f_num
        self.PC=pc
        self.PM=pm
        self.YITAL1=yital1
        self.YITAL2=yital2

    def initial_population(self):
        """
            # 初始化个体
        :return:
        """
        population_np=(self.X_MAX_ARRAY - self.X_MIN_ARRAY) * np.random.random_sample((self.POP_SIZE,self.X_NUM)) + self.X_MIN_ARRAY
        fitness_np=self.calculate_fitness(population_np)
        nnd_np=np.full(self.POP_SIZE, 0)
        rank_np = np.full(self.POP_SIZE, 0)
        popu=Population(population_np,fitness_np,nnd_np,rank_np)
        return popu

    def create_population(self,population_np):
        """
            # 初始化个体
        :return:
        """
        fitness_np=self.calculate_fitness(population_np)
        nnd_np=np.full(self.POP_SIZE, 0)
        rank_np = np.full(self.POP_SIZE, 0)
        popu=Population(population_np,fitness_np,nnd_np,rank_np)
        return popu




    def calculate_fitness(self,population_np):
        if (self.FITNESS_NAME=='ZDT1'):
            f1=float(population_np[:,0])
            sum1 = np.array([np.sum(population_np[i, 1:]) for i in range(len(population_np[:, 0]))])  # 因为传入的是整个population
            g = (1 + 9 * (sum1 / (self.X_NUM - 1))).astype(np.float)
            f2 = g * (1 - np.power(population_np[:, 0] / g, 0.5))
            fitness_np=np.hstack((f1.reshape(self.POP_SIZE,1), f2.reshape(self.POP_SIZE,1)))
        elif (self.FITNESS_NAME=='ZDT2'):
            f1=float(population_np[0])
            sum1 = np.array([np.sum(population_np[i, 1:]) for i in range(len(population_np[:, 0]))])  # 因为传入的是整个population
            g=float(1+9*(sum1/(self.X_NUM-1))).astype(np.float)
            f2=g*(1-(f1/g)**2)
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1)))
        elif (self.FITNESS_NAME=='ZDT3'):
            f1=float(population_np[0])
            sum1 = np.array([np.sum(population_np[i, 1:]) for i in range(len(population_np[:, 0]))])  # 因为传入的是整个population
            g = (1 + 9 * (sum1 / (self.X_NUM - 1))).astype(np.float)
            h = 1 - (f1 / g) ** 0.5 - (f1 / g) * np.sin(10 * np.pi * f1)
            f2 = g * h
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1)))
        elif (self.FITNESS_NAME=='ZDT4'):
            f1 = float(population_np[0])
            sum1 = np.array(
                [np.sum(np.power(population_np[i, 1:], 2) - 10 * np.cos(4 * np.pi * population_np[i, 1:])) for i in
                 range(len(population_np[:, 0]))])  # 因为传入的是整个population
            g = (91 + sum1).astype(np.float)
            f2 = (g * (1 - (population_np[:, 0] / g) ** 0.5)).astype(np.float)
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1)))
        elif (self.FITNESS_NAME=='ZDT6'):
            f1 = (1 - np.exp(-4 * population_np[:, 0]) * (np.sin(6 * np.pi * population_np[:, 0])) ** 6).astype(np.float)
            sum1 = np.array([np.sum(population_np[i, 1:]) for i in range(len(population_np[:, 0]))])  # 因为传入的是整个population
            g = (1 + 9 * ((sum1 / (10 - 1)) ** 0.25)).astype(np.float)
            f1 = (1 - np.exp(-4 * population_np[:, 0]) * (np.sin(6 * np.pi * population_np[:, 0])) ** 6).astype(np.float)
            f2 = g * (1 - (f1 / g) ** 2)
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1)))
        elif (self.FITNESS_NAME=='DTLZ1'):
            sum1 =  np.array([np.sum(np.power(population_np[i,2:] - 0.5, 2) - np.cos(20 * np.pi * (population_np[i,2:] - 0.5))) for i in range(len(population_np[:, 0]))])
            g = (100 * (x_num - 2) + 100 * sum1).astype(np.float)
            f1 = ((1 + g) * population_np[:,0] * population_np[:,1]).astype(np.float)
            f2 = ((1 + g) * population_np[:,0] * (1 - population_np[:,1])).astype(np.float)
            f3 = ((1 + g) * (1 - population_np[:,0])).astype(np.float)
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1), f3.reshape(self.POP_SIZE, 1)))
        elif (self.FITNESS_NAME=='DTLZ2'):
            g = np.array([np.sum(population_np[i,2:] ** 2) for i in range(len(population_np[:, 0]))])
            f1 = ((1 + g) * math.cos(0.5 * math.pi * population_np[:,0]) * math.cos(0.5 * math.pi * population_np[:,1])).astype(np.float)
            f2 = ((1 + g) * math.cos(0.5 * math.pi * population_np[:,0]) * math.sin(0.5 * math.pi * population_np[:,1])).astype(np.float)
            f3 = ((1 + g) * math.sin(0.5 * math.pi * population_np[:,0])).astype(np.float)
            fitness_np = np.hstack((f1.reshape(self.POP_SIZE, 1), f2.reshape(self.POP_SIZE, 1), f3.reshape(self.POP_SIZE, 1)))

    # 非支配排序
    def non_domination_sort(self,popu):
        #non_domination_sort 初始种群的非支配排序和计算拥挤度
        #初始化pareto等级为1
        pareto_rank=1
        fronts={}#初始化一个字典
        fronts[pareto_rank]=[]#pareto等级为pareto_rank的集合
        pn={}
        ps={}
        for i in range(self.POP_SIZE):
            #计算出种群中每个个体p的被支配个数n和该个体支配的解的集合s
            pn[i]=0#被支配个体数目n
            ps[i]=[]#支配解的集合s
            for j in range(self.POP_SIZE):
                less=0#y'的目标函数值小于个体的目标函数值数目
                equal=0#y'的目标函数值等于个体的目标函数值数目
                greater=0#y'的目标函数值大于个体的目标函数值数目
                for k in range(self.F_NUM):
                    if (popu.fitness_np[i,k]<popu.fitness_np[j,k]):
                        less=less+1
                    elif (popu.fitness_np[i,k] == popu.fitness_np[j,k]):
                        equal=equal+1
                    else:
                        greater=greater+1
                if (less==0 and equal!=self.F_NUM):
                    pn[i]=pn[i]+1
                elif (greater==0 and equal!=self.F_NUM):
                    ps[i].append(j)
            if (pn[i]==0):
                # Individual对象 有四个属性 chromosome,nnd,paretorank,f
                popu.rank_np[i]=1#储存个体的等级信息
                fronts[pareto_rank].append(i)
        #求pareto等级为pareto_rank+1的个体
        while (len(fronts[pareto_rank])!=0):
            temp=[]
            for i in range(len(fronts[pareto_rank])):
                if (len(ps[fronts[pareto_rank][i]])!=0):
                    for j in range(len(ps[fronts[pareto_rank][i]])):
                        pn[ps[fronts[pareto_rank][i]][j]]=pn[ps[fronts[pareto_rank][i]][j]]-1#nl=nl-1
                        if pn[ps[fronts[pareto_rank][i]][j]]==0:
                            popu.rank_np[ps[fronts[pareto_rank][i]][j]]= pareto_rank + 1#储存个体的等级信息
                            temp.append(ps[fronts[pareto_rank][i]][j])
            pareto_rank=pareto_rank+1
            fronts[pareto_rank]=temp
        return fronts, popu

    def sorted_population_by_rank(self,popu):
        sorted_popu=copy.deepcopy(popu)
        temp_sorting_rank_np=copy.deepcopy(popu.rank_np)   #临时存储变量，这样不用改变popu的属性
        for i in range(self.POP_SIZE):
            min_index=np.where(temp_sorting_rank_np == np.min(temp_sorting_rank_np))[0]
            sorted_popu.population_np[i]=popu.population_np[min_index]
            sorted_popu.fitness_np[i]=popu.fitness_np[min_index]
            sorted_popu.rank_np[i]=popu.rank_np[min_index]
            sorted_popu.nnd_np[i]=popu.nnd_np[min_index]
            temp_sorting_rank_np[min_index]=88888
        return sorted_popu

    def sorted_population_by_fitness(self, y_population_np_list, y_fitness_np_list, f_num_id):
        sorted_y_population_np_list = copy.deepcopy(y_population_np_list)
        sorted_y_fitness_np_list = copy.deepcopy(y_fitness_np_list)
        temp_y_fitness_np_list = copy.deepcopy(y_fitness_np_list)
        temp_y_fitness_np_array=np.array(temp_y_fitness_np_list)
        for i in range(len(y_population_np_list)):
            min_index=np.where(temp_y_fitness_np_array[:,f_num_id] == np.min(temp_y_fitness_np_array[:,f_num_id]))[0]
            sorted_y_population_np_list[i] = y_population_np_list[min_index]
            sorted_y_fitness_np_list[i] = y_fitness_np_list[min_index]
            temp_y_fitness_np_array[min_index,f_num_id] = 88888
        return sorted_y_population_np_list,sorted_y_fitness_np_list


    # 拥挤度排序
    def crowding_distance_sort(self,fronts,popu):
        #计算拥挤度
        crowd_population_np=[]
        sorted_popu=self.sorted_population_by_rank(popu)#按照pareto等级排序后种群，升序排列
        index1=[]
        for i in range(self.POP_SIZE):
            index1.append(popu.population_np.index(sorted_popu.population_np[i]))
        #对于每个等级的个体开始计算拥挤度
        current_index = 0
        for pareto_rank in range(len(fronts) - 1):#计算F的循环时多了一次空，所以减掉,由于pareto从1开始，再减一次
            nd=np.zeros(len(fronts[pareto_rank + 1]))#拥挤度初始化为0
            y_population_np_list=[]#储存当前处理的等级的个体
            y_fitness_list=[]
            yF=np.zeros((len(fronts[pareto_rank + 1]), self.F_NUM))
            for i in range(len(fronts[pareto_rank + 1])):
                y_population_np_list.append(sorted_popu.population_np[current_index + i])
                y_fitness_list.append(sorted_popu.fitness_np[current_index + i])
            current_index=current_index + i + 1
            #对于每一个目标函数fm
            for f_num_id in range(self.F_NUM):
                #根据该目标函数值对该等级的个体进行排序
                index_objective=[]#通过目标函数排序后的个体索引
                objective_sort_y_population_np,objective_y_fitness_np=self.sorted_population_by_fitness(y_population_np_list,y_fitness_list,f_num_id)#通过目标函数排序后的个体
                for j in range(len(objective_sort_y_population_np)):
                    index_objective.append(y_population_np_list.index(objective_sort_y_population_np[j]))
                #记fmax为最大值，fmin为最小值
                fmin=objective_y_fitness_np[0,f_num_id]
                fmax=objective_y_fitness_np[-1,f_num_id]
                #对排序后的两个边界拥挤度设为1d和nd设为无穷
                yF[index_objective[0]][f_num_id]=float("inf")
                yF[index_objective[len(index_objective)-1]][f_num_id]=float("inf")
                #计算nd=nd+(fm(i+1)-fm(i-1))/(fmax-fmin)
                j=1
                while (j<=(len(index_objective)-2)):
                    pre_f=objective_y_fitness_np[j-1,f_num_id]
                    next_f=objective_y_fitness_np[j+1,f_num_id]
                    if (fmax-fmin==0):
                        yF[index_objective[j]][f_num_id]=float("inf")
                    else:
                        yF[index_objective[j]][f_num_id]=float((next_f-pre_f)/(fmax-fmin))
                    j=j+1
            #多个目标函数拥挤度求和
            nd=np.sum(yF,axis=1)
            for i in range(len(y_population_np_list)):
                nnd_np[i]=nd[i]
                crowd_population_np.append(y_population_np_list[i])
        return crowd_population_np

    # 锦标竞赛选择
    def tournament_selection2(self,population):
        touranment=2
        half_pop_size=round(self.POP_SIZE / 2)
        chromo_candidate=np.zeros(touranment)
        chromo_rank=np.zeros(touranment)
        chromo_distance=np.zeros(touranment)
        chromo_parent=[]
        for i in range(half_pop_size):
            for j in range(touranment):
                chromo_candidate[j]=round(self.POP_SIZE * random.random())
                if chromo_candidate[j]==self.POP_SIZE:#索引不能为N
                    chromo_candidate[j]= self.POP_SIZE - 1
            while (chromo_candidate[0] == chromo_candidate[1]):
                chromo_candidate[0]=round(self.POP_SIZE * random.random())
                if chromo_candidate[0]==self.POP_SIZE:
                    chromo_candidate[0]= self.POP_SIZE - 1
            chromo_rank[0]=population[int(chromo_candidate[0])].paretorank
            chromo_rank[1]=population[int(chromo_candidate[1])].paretorank
            chromo_distance[0]=population[int(chromo_candidate[0])].nnd
            chromo_distance[1]=population[int(chromo_candidate[1])].nnd
            #取出低等级的个体索引
            minchromo_candidate=np.argmin(chromo_rank)
            #多个索引按拥挤度排序
            if (chromo_rank[0]==chromo_rank[1]):
                maxchromo_candidate=np.argmax(chromo_distance)
                chromo_parent.append(population[maxchromo_candidate])
            else:
                chromo_parent.append(population[minchromo_candidate])
        return chromo_parent




    def crossover_and_mutation_nsga3(self,pop):
        pop1 = copy.deepcopy(pop[0:int(pop.shape[0] / 2), :])
        pop2 = copy.deepcopy(pop[(int(pop.shape[0] / 2)):(int(pop.shape[0] / 2) * 2), :])
        shape_row, shape_column = pop1.shape[0], pop1.shape[1]
        # 模拟二进制交叉
        beta = np.zeros((shape_row, shape_column))
        mu = np.random.random_sample([shape_row, shape_column])
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (yita1 + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (yita1 + 1))
        beta = beta * ((-1) ** (np.random.randint(2, size=(shape_row, shape_column))))
        beta[np.random.random_sample([shape_row, shape_column]) < 0.5] = 1
        beta[np.tile(np.random.random_sample([shape_row, 1]) > self.PC, (1, shape_column))] = 1
        off = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))
        # 多项式变异
        low = np.zeros((2 * shape_row, shape_column))   #确定上下边界
        up = np.ones((2 * shape_row, shape_column))   #确定上下边界
        site = np.random.random_sample([2 * shape_row, shape_column]) < self.PM / shape_column
        mu = np.random.random_sample([2 * shape_row, shape_column])
        temp = site & (mu <= 0.5)
        off[off < low] = low[off < low]
        off[off > up] = up[off > up]
        off[temp] = off[temp] + (up[temp] - low[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                    (1 - (off[temp] - low[temp]) / (up[temp] - low[temp])) ** (yita2 + 1))) ** (1 / (yita2 + 1)) - 1)
        temp = site & (mu > 0.5)
        off[temp] = off[temp] + (up[temp] - low[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                    (1 - (up[temp] - off[temp]) / (up[temp] - low[temp])) ** (yita2 + 1))) ** (1 / (yita2 + 1)))

        return off

    def elitism(self,combine_chromo2):
        chromo=[]
        index1=0
        index2=0
        #根据pareto等级从高到低进行排序
        chromo_rank=sorted(combine_chromo2, key=lambda Individual:Individual.paretorank)
        flag=chromo_rank[self.POP_SIZE-1].paretorank
        for i in range(len(chromo_rank)):
            if (chromo_rank[i].paretorank==(flag)):
                index1=i
                break
            else:
                chromo.append(chromo_rank[i])
        for i in range(len(chromo_rank)):
            if (chromo_rank[i].paretorank==(flag + 1)):
                index2=i
                break
        temp=[]
        aaa=index1
        if (index2==0):
            index2=len(chromo_rank)
        while (aaa<index2):
            temp.append(chromo_rank[aaa])
            aaa=aaa+1
        temp_crowd=sorted(temp, key=lambda Individual:Individual.paretorank, reverse=True)
        remainN=self.POP_SIZE-index1
        for i in range(remainN):
            chromo.append(temp_crowd[i])
        return chromo



    # 判别支配关系
    def dominate(self,y1, y2):
        less=0#y1的目标函数值小于y2个体的目标函数值数目
        equal=0#y1的目标函数值等于y2个体的目标函数值数目
        greater=0#y1的目标函数值大于y2个体的目标函数值数目
        for i in range(len(y1)):
            if y1[i]>y2.fitness_nsga2[i]:
                greater=greater+1
            elif y1[i]==y2.fitness_nsga2[i]:
                equal=equal+1
            else:
                less=less+1
        if(greater==0 and equal!=len(y1)):
            return True#y1支配y2返回正确
        elif(less==0 and equal!=len(y1)):
            return False#y2支配y1返回false
        else:
            return None


    def plot_show(self,current_population):
        # ------------------------画图对比--------------------------
        x = []
        y = []
        z = []
        if f_num == 2:
            for i in range(len(current_population)):
                x.append(current_population[i].fitness_nsga2[0])
                y.append(current_population[i].fitness_nsga2[1])
            plt.scatter(x, y, marker='o', color='red', s=40)
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.show()
        elif f_num == 3:
            for i in range(len(current_population)):
                x.append(current_population[i].fitness_nsga2[0])
                y.append(current_population[i].fitness_nsga2[1])
                z.append(current_population[i].fitness_nsga2[2])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x, y, z, c='r')
            plt.show()


    def coverage_and_distance_evaluation(self,current_population):
        # --------------------Coverage(C-metric)---------------------
        A = fun_txt
        B = current_population
        number = 0
        for i in range(len(B)):
            nn = 0
            for j in range(len(A)):
                if (self.dominate(A[j], B[i])):
                    nn = nn + 1  # B[i]被A支配的个体数目+1
            if (nn != 0):
                number = number + 1
        C_AB = float(number / len(B))
        print("C_AB：%2f" % C_AB)
        # -----Distance from Representatives in the PF(D-metric)-----
        A = current_population
        P = fun_txt
        min_d = 0
        for i in range(len(P)):
            temp = []
            for j in range(len(A)):
                dd = 0
                for k in range(self.F_NUM):
                    dd = dd + float((P[i][k] - A[j].fitness_nsga2[k]) ** 2)
                temp.append(math.sqrt(dd))
            min_d = min_d + np.min(temp)
        D_AP = float(min_d / len(P))
        print("D_AP：%2f" % D_AP)

    def nsga2_main(self):
        # ------------------------初始化种群--------------------------
        popu = self.initial_population()
        # -----------------初始化种群的非支配排序----------------------
        fronts, popu =  self.non_domination_sort(popu)  # fronts为pareto等级为pareto_rank的集合(包括每个个体p的被支配个数n和该个体支配的解的集合s),population_non最后一列加入个体的等级
        # --------------------计算拥挤度进行排序-----------------------
        popu =  self.crowding_distance_sort(fronts,popu)
        # ------------------------迭代更新--------------------------
        gen = 1
        while (gen <= MAX_GEN):
            print(gen)
            for i in range(POP_SIZE):
                ##二进制竞赛选择(k取了pop/2，所以选两次)
                population_parent_1 =  self.tournament_selection2(current_population_np)
                population_parent_2 =  self.tournament_selection2(current_population_np)
                population_parent = population_parent_1 + population_parent_2
                ##模拟二进制交叉与多项式变异
                population_offspring =  self.crossover_and_mutation_nsga3(population_parent)
                ##精英保留策略
                # 将父代和子代合并
                double_population_np = current_population_np + population_offspring
                double_fitness_np=self.calculate_fitness(double_population_np)
                double_nnd_np = np.full(self.POP_SIZE, 0)
                double_rank_np = np.full(self.POP_SIZE, 0)
                # 快速非支配排序
                F2, combine_population_1 =  self.non_domination_sort( double_population_np,double_fitness_np,double_nnd_np,double_rank_np)
                # 计算拥挤度进行排序
                combine_population_2 =  self.crowding_distance_sort(combine_population_1,F2)
                # 精英保留产生下一代种群
                current_population_np =  self.elitism(combine_population_2)
            if (gen % 10) == 0:
                print("%d gen has completed!\n" % gen)
            gen = gen + 1;
        end = time.time()
        print("循环时间：%2f秒" % (end - start))
        return current_population_np

if __name__=="__main__":
    #------------------------参数输入--------------------------
    POP_SIZE=136#种群规模
    fitness_function_name= 'DTLZ1'#测试函数DTLZ2
    # 获取评价函数的元信息
    f_num, x_num, x_min_array, x_max_array, fun_txt=fitness_fun_mate_infor(fitness_function_name)
    MAX_GEN=10#最大进化代数
    CROSSOVER_PROBABILITY=0.9#交叉概率
    MUTATION_PROBABILITY= 1 / x_num#变异概率
    yita1=20#模拟二进制交叉参数2
    yita2=20#多项式变异参数5

    nsga2_obj=NSGA2(POP_SIZE, fitness_function_name, x_num, x_min_array, x_max_array, f_num, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, yita1, yita2)
    # 主函数
    current_population=nsga2_obj.nsga2_main()
    # 制图表达
    nsga2_obj.plot_show(current_population)
    # 评价
    nsga2_obj.coverage_and_distance_evaluation(current_population)