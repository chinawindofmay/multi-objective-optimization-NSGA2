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

def fitness_fun_infor(fitness_function_name):
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


# Individual对象 有四个属性 chromosome,nnd,paretorank,fitness_fun
class Individual():
    def __init__(self, chromosome, fitness_function_name, x_num):
        self.chromosome=chromosome
        self.nnd=0
        self.paretorank=0
        if (fitness_function_name=='ZDT1'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2=g*(1-(f1/g)**(0.5))
            self.fitness=[f1, f2]
        elif (fitness_function_name=='ZDT2'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g=float(1+9*(sum1/(x_num-1)))
            f2=g*(1-(f1/g)**2)
            self.fitness=[f1, f2]
        elif (fitness_function_name=='ZDT3'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            h = 1 - (f1 / g) ** 0.5 - (f1 / g) * math.sin(10 * math.pi * f1)
            f2 = g * h
            self.fitness=[f1, f2]
        elif (fitness_function_name=='ZDT4'):
            f1 = float(chromosome[0])
            sum2 = np.sum(np.power(chromosome[1:], 2) - 10 * np.cos(4 * np.pi * chromosome[1:]))
            g = float(91 + sum2)
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.fitness=[f1, f2]
        elif (fitness_function_name=='ZDT6'):
            f1 = float(1 - math.exp(-4 * chromosome[0]) * (math.sin(6 * math.pi * chromosome[0])) ** 6)
            sum1 = np.sum(chromosome[1:])
            g = 1 + 9 * ((sum1 / (x_num - 1)) ** 0.25)
            f2 = g * (1 - (f1 / g) ** 2)
            self.fitness=[f1, f2]
        elif (fitness_function_name=='DTLZ1'):
            sum1 = np.sum(np.power(chromosome[2:] - 0.5, 2) - np.cos(20 * np.pi * (chromosome[2:] - 0.5)))
            g = float(100 * (x_num - 2) + 100 * sum1)
            f1 = float((1 + g) * chromosome[0] * chromosome[1])
            f2 = float((1 + g) * chromosome[0] * (1 - chromosome[1]))
            f3 = float((1 + g) * (1 - chromosome[0]))
            self.fitness=[f1, f2, f3]
        elif (fitness_function_name=='DTLZ2'):
            g = np.sum(chromosome[2:] ** 2)
            f1 = float((1 + g) * math.cos(0.5 * math.pi * chromosome[0]) * math.cos(0.5 * math.pi * chromosome[1]))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * chromosome[0]) * math.sin(0.5 * math.pi * chromosome[1]))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * chromosome[0]))
            self.fitness=[f1, f2, f3]

# 初始化个体
def initial_population(POP_SIZE, fitness_function_name, x_num, X_min_array, X_max_array, f_num):
    population=[]
    #种群初始化以及产生lamda
    for i in range(POP_SIZE):
        chromosome = (X_max_array - X_min_array) * np.random.random_sample(x_num) + X_min_array
        population.append(Individual(chromosome,fitness_function_name,x_num))
    return population

# 非支配排序
def non_domination_sort(population, pop_size,  f_num, x_num):
    #non_domination_sort 初始种群的非支配排序和计算拥挤度
    #初始化pareto等级为1
    pareto_rank=1
    fronts={}#初始化一个字典
    fronts[pareto_rank]=[]#pareto等级为pareto_rank的集合
    pn={}
    ps={}
    for i in range(pop_size):
        #计算出种群中每个个体p的被支配个数n和该个体支配的解的集合s
        pn[i]=0#被支配个体数目n
        ps[i]=[]#支配解的集合s
        for j in range(pop_size):
            less=0#y'的目标函数值小于个体的目标函数值数目
            equal=0#y'的目标函数值等于个体的目标函数值数目
            greater=0#y'的目标函数值大于个体的目标函数值数目
            for k in range(f_num):
                if (population[i].fitness[k]<population[j].fitness[k]):
                    less=less+1
                elif (population[i].fitness[k] == population[j].fitness[k]):
                    equal=equal+1
                else:
                    greater=greater+1
            if (less==0 and equal!=f_num):
                pn[i]=pn[i]+1
            elif (greater==0 and equal!=f_num):
                ps[i].append(j)
        if (pn[i]==0):
            # Individual对象 有四个属性 chromosome,nnd,paretorank,f
            population[i].paretorank=1#储存个体的等级信息
            fronts[pareto_rank].append(i)
    #求pareto等级为pareto_rank+1的个体
    while (len(fronts[pareto_rank])!=0):
        temp=[]
        for i in range(len(fronts[pareto_rank])):
            if (len(ps[fronts[pareto_rank][i]])!=0):
                for j in range(len(ps[fronts[pareto_rank][i]])):
                    pn[ps[fronts[pareto_rank][i]][j]]=pn[ps[fronts[pareto_rank][i]][j]]-1#nl=nl-1
                    if pn[ps[fronts[pareto_rank][i]][j]]==0:
                        population[ps[fronts[pareto_rank][i]][j]].paretorank= pareto_rank + 1#储存个体的等级信息
                        temp.append(ps[fronts[pareto_rank][i]][j])
        pareto_rank=pareto_rank+1
        fronts[pareto_rank]=temp
    return fronts, population

# 拥挤度排序
def crowding_distance_sort( population_non,fronts, f_num, x_num):
    #计算拥挤度
    ppp=[]
    #按照pareto等级对种群中的个体进行排序
    temp=sorted(population_non, key=lambda Individual:Individual.paretorank)#按照pareto等级排序后种群
    index1=[]
    for i in range(len(temp)):
        index1.append(population_non.index(temp[i]))
    #对于每个等级的个体开始计算拥挤度
    current_index = 0
    for pareto_rank in range(len(fronts) - 1):#计算F的循环时多了一次空，所以减掉,由于pareto从1开始，再减一次
        nd=np.zeros(len(fronts[pareto_rank + 1]))#拥挤度初始化为0
        y=[]#储存当前处理的等级的个体
        yF=np.zeros((len(fronts[pareto_rank + 1]), f_num))
        for i in range(len(fronts[pareto_rank + 1])):
            y.append(temp[current_index + i])
        current_index=current_index + i + 1
        #对于每一个目标函数fm
        for i in range(f_num):
            #根据该目标函数值对该等级的个体进行排序
            index_objective=[]#通过目标函数排序后的个体索引
            objective_sort=sorted(y, key=lambda Individual:Individual.fitness[i])#通过目标函数排序后的个体
            for j in range(len(objective_sort)):
                index_objective.append(y.index(objective_sort[j]))
            #记fmax为最大值，fmin为最小值
            fmin=objective_sort[0].fitness[i]
            fmax=objective_sort[len(objective_sort)-1].fitness[i]
            #对排序后的两个边界拥挤度设为1d和nd设为无穷
            yF[index_objective[0]][i]=float("inf")
            yF[index_objective[len(index_objective)-1]][i]=float("inf")
            #计算nd=nd+(fm(i+1)-fm(i-1))/(fmax-fmin)
            j=1
            while (j<=(len(index_objective)-2)):
                pre_f=objective_sort[j-1].fitness[i]
                next_f=objective_sort[j+1].fitness[i]
                if (fmax-fmin==0):
                    yF[index_objective[j]][i]=float("inf")
                else:
                    yF[index_objective[j]][i]=float((next_f-pre_f)/(fmax-fmin))
                j=j+1
        #多个目标函数拥挤度求和
        nd=np.sum(yF,axis=1)
        for i in range(len(y)):
            y[i].nnd=nd[i]
            ppp.append(y[i])
    return ppp

# 锦标竞赛选择
def tournament_selection2(population, POP_SIZE):
    touranment=2
    half_pop_size=round(POP_SIZE / 2)
    chromo_candidate=np.zeros(touranment)
    chromo_rank=np.zeros(touranment)
    chromo_distance=np.zeros(touranment)
    chromo_parent=[]
    for i in range(half_pop_size):
        for j in range(touranment):
            chromo_candidate[j]=round(POP_SIZE * random.random())
            if chromo_candidate[j]==POP_SIZE:#索引不能为N
                chromo_candidate[j]= POP_SIZE - 1
        while (chromo_candidate[0] == chromo_candidate[1]):
            chromo_candidate[0]=round(POP_SIZE * random.random())
            if chromo_candidate[0]==POP_SIZE:
                chromo_candidate[0]= POP_SIZE - 1
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

# def crossover_and_mutation_nsga3(population_parent, POP_SIZE, x_num, pc, pm, yita1, yita2):
#     #为了便于采用numpy广播机制计算，首先将chrome_parent转换为pop
#     pop=np.full((POP_SIZE,x_num),0.001)
#     for i in range(POP_SIZE):
#         individual=population_parent[i]
#         pop[i,:]=individual.chromosome
#
#     pop1 = copy.deepcopy(pop[0:int(pop.shape[0] / 2), :])
#     pop2 = copy.deepcopy(pop[(int(pop.shape[0] / 2)):(int(pop.shape[0] / 2) * 2), :])
#     shape_row, shape_column = pop1.shape[0], pop1.shape[1]
#     # 模拟二进制交叉
#     beta = np.zeros((shape_row, shape_column))
#     mu = np.random.random_sample([shape_row, shape_column])
#     beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (yita1 + 1))
#     beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (yita1 + 1))
#     beta = beta * ((-1) ** (np.random.randint(2, size=(shape_row, shape_column))))
#     beta[np.random.random_sample([shape_row, shape_column]) < 0.5] = 1
#     beta[np.tile(np.random.random_sample([shape_row, 1]) > pc, (1, shape_column))] = 1
#     off = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))
#     # 多项式变异
#     low = np.zeros((2 * shape_row, shape_column))   #确定上下边界
#     up = np.ones((2 * shape_row, shape_column))   #确定上下边界
#     site = np.random.random_sample([2 * shape_row, shape_column]) < pm / shape_column
#     mu = np.random.random_sample([2 * shape_row, shape_column])
#     temp = site & (mu <= 0.5)
#     off[off < low] = low[off < low]
#     off[off > up] = up[off > up]
#     off[temp] = off[temp] + (up[temp] - low[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
#                 (1 - (off[temp] - low[temp]) / (up[temp] - low[temp])) ** (yita2 + 1))) ** (1 / (yita2 + 1)) - 1)
#     temp = site & (mu > 0.5)
#     off[temp] = off[temp] + (up[temp] - low[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
#                 (1 - (up[temp] - off[temp]) / (up[temp] - low[temp])) ** (yita2 + 1))) ** (1 / (yita2 + 1)))
#
#     # 将pop转换为chrome_parent
#     off_spring=[]
#     for i in range(POP_SIZE):
#         off_spring.append(Individual(off[i,:],fitness_function_name,x_num))
#     return off_spring

def cross_mutation(population_parent,x_num, x_min, x_max, pc, pm, yita1, yita2,fitness_function_name):
    #模拟二进制交叉和多项式变异
    ###模拟二进制交叉
    population_offspring=[]
    #随机选取两个父代个体
    for i in range(round(len(population_parent) / 2)):
        parent_1=round(len(population_parent) * random.random())
        if (parent_1==len(population_parent)):
            parent_1= len(population_parent) - 1
        parent_2=round(len(population_parent) * random.random())
        if (parent_2==len(population_parent)):
            parent_2= len(population_parent) - 1
        while (parent_1==parent_2):
            parent_1=round(len(population_parent) * random.random())
            if (parent_1==len(population_parent)):
                parent_1= len(population_parent) - 1
        individual_parent1=population_parent[parent_1]
        individual_parent2=population_parent[parent_2]
        individual_off1=individual_parent1
        individual_off2=individual_parent2
        if(random.random()<pc):
            #初始化子代种群
            off1x=[]
            off2x=[]
            #模拟二进制交叉
            for j in range(x_num):
                u1=random.random()
                if(u1<=0.5):
                    gama=float((2*u1)**(1/(yita1+1)))
                else:
                    gama=float((1/(2*(1-u1)))**(1/(yita1+1)))
                off11=float(0.5*((1+gama)*individual_parent1.chromosome[j]+(1-gama)*individual_parent2.chromosome[j]))
                off22=float(0.5*((1-gama)*individual_parent1.chromosome[j]+(1+gama)*individual_parent2.chromosome[j]))
                #使子代在定义域内
                if (off11>x_max[j]):
                    off11=x_max[j]
                elif (off11<x_min[j]):
                    off11=x_min[j]
                if (off22>x_max[j]):
                    off22=x_max[j]
                elif (off22<x_min[j]):
                    off22=x_min[j]
                off1x.append(off11)
                off2x.append(off22)
            individual_off1=Individual(np.array(off1x),fitness_function_name,x_num)
            individual_off2=Individual(np.array(off2x),fitness_function_name,x_num)
        #多项式变异
        if (random.random()<pm):
            off1x=[]
            off2x=[]
            for j in range(x_num):
                low = x_min[j]  # 确定上下边界
                up = x_max[j]  # 确定上下边界
                delta1 = polynomial_mutation(low, up, individual_off1.chromosome[j],yita2)
                off11=float(individual_off1.chromosome[j]+delta1)
                delta2 = polynomial_mutation(low, up, individual_off2.chromosome[j], yita2)
                off22=float(individual_off2.chromosome[j]+delta2)
                if (off11>x_max[j]):
                    off11=x_max[j]
                elif (off11<x_min[j]):
                    off11=x_min[j]
                if (off22>x_max[j]):
                    off22=x_max[j]
                elif (off22<x_min[j]):
                    off22=x_min[j]
                off1x.append(off11)
                off2x.append(off22)
            individual_off1=Individual(np.array(off1x),fitness_function_name,x_num)
            individual_off2=Individual(np.array(off2x),fitness_function_name,x_num)
        population_offspring.append(individual_off1)
        population_offspring.append(individual_off2)
    return population_offspring

# 多项式变异
def polynomial_mutation(low, up, x,yita2):
    dd1_a = (x - low) / (up - low)
    dd1_b = (up - x) / (up - low)
    if dd1_a < dd1_b:
        dd1 = dd1_a
    else:
        dd1 = dd1_b
    dd1 = (1 - dd1) ** (yita2 + 1)
    u1 = random.random()
    if (u1 < 0.5):
        delta1 = (up - low) * ((2 * u1 + (1 - 2 * u1) * dd1) ** (1 / (yita2 + 1)) - 1)
    else:
        delta1 = (up - low) * (
                1 - (2 * (1 - u1) + 2 * (u1 - 0.5) * dd1) ** (1 / (yita2 + 1)))
    return delta1


def elitism(N,combine_chromo2,f_num,x_num):
    chromo=[]
    index1=0
    index2=0
    #根据pareto等级从高到低进行排序
    chromo_rank=sorted(combine_chromo2, key=lambda Individual:Individual.paretorank)
    flag=chromo_rank[N-1].paretorank
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
    remainN=N-index1
    for i in range(remainN):
        chromo.append(temp_crowd[i])
    return chromo
        


# 判别支配关系
def dominate(y1, y2):
    less=0#y1的目标函数值小于y2个体的目标函数值数目
    equal=0#y1的目标函数值等于y2个体的目标函数值数目
    greater=0#y1的目标函数值大于y2个体的目标函数值数目
    for i in range(len(y1)):
        if y1[i]>y2.fitness[i]:
            greater=greater+1
        elif y1[i]==y2.fitness[i]:
            equal=equal+1
        else:
            less=less+1
    if(greater==0 and equal!=len(y1)):
        return True#y1支配y2返回正确
    elif(less==0 and equal!=len(y1)):
        return False#y2支配y1返回false
    else:
        return None

if __name__=="__main__":
    #------------------------参数输入--------------------------
    POP_SIZE=150#种群规模
    # fitness_function_name= 'DTLZ1'#测试函数DTLZ2
    fitness_function_name= 'ZDT3'#测试函数DTLZ2
    F_NUM, X_COUNT, X_min_array, X_max_array, fun_txt=fitness_fun_infor(fitness_function_name)
    MAX_GEN=200#最大进化代数
    CROSSOVER_PROBABILITY=1#交叉概率
    MUTATION_PROBABILITY= 1 / X_COUNT#变异概率
    yita1=20#模拟二进制交叉参数2
    yita2=20#多项式变异参数5
    #------------------------初始化种群--------------------------
    current_population=initial_population(POP_SIZE, fitness_function_name, X_COUNT, X_min_array, X_max_array, F_NUM)
    #-----------------初始化种群的非支配排序----------------------
    fronts, population_non=non_domination_sort(current_population,POP_SIZE,  F_NUM, X_COUNT)     #fronts为pareto等级为pareto_rank的集合(包括每个个体p的被支配个数n和该个体支配的解的集合s),population_non最后一列加入个体的等级
    #--------------------计算拥挤度进行排序-----------------------
    current_population=crowding_distance_sort(population_non, fronts, F_NUM, X_COUNT)
    #------------------------迭代更新--------------------------
    gen=1
    while(gen<=MAX_GEN):
        print(gen)
        for i in range(POP_SIZE):
            ##二进制竞赛选择(k取了pop/2，所以选两次)
            population_parent_1 = tournament_selection2(current_population, POP_SIZE)
            population_parent_2 = tournament_selection2(current_population, POP_SIZE)
            population_parent= population_parent_1 + population_parent_2
            ##模拟二进制交叉与多项式变异
            # population_offspring=cross_mutation(population_parent, F_NUM, X_COUNT, X_min_array, X_max_array, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, yita1, yita2, fitness_function_name)
            population_offspring=cross_mutation(population_parent, X_COUNT, X_min_array, X_max_array, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, yita1, yita2,fitness_function_name)
            # population_offspring=crossover_and_mutation_nsga3(population_parent, POP_SIZE, X_COUNT, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, yita1, yita2)
            ##精英保留策略
            #将父代和子代合并
            double_population= current_population + population_offspring
            #快速非支配排序
            F2, combine_population_1=non_domination_sort( double_population,len(double_population), F_NUM, X_COUNT)
            #计算拥挤度进行排序
            combine_population_2=crowding_distance_sort( combine_population_1,F2, F_NUM, X_COUNT)
            #精英保留产生下一代种群
            current_population=elitism(POP_SIZE, combine_population_2, F_NUM, X_COUNT)
        if (gen%10) == 0:
            print("%d gen has completed!\n"%gen)
        gen=gen+1;
    end=time.time()
    print("循环时间：%2f秒"%(end-start))
    #------------------------画图对比--------------------------
    x=[]
    y=[]
    z=[]
    if F_NUM==2:
        for i in range(len(current_population)):
            x.append(current_population[i].fitness[0])
            y.append(current_population[i].fitness[1])
        plt.scatter(x,y,marker='o',color='red',s=40)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.show()
    elif F_NUM==3:
        for i in range(len(current_population)):
            x.append(current_population[i].fitness[0])
            y.append(current_population[i].fitness[1])
            z.append(current_population[i].fitness[2])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x,y,z,c='r')
        plt.show()

    #--------------------Coverage(C-metric)---------------------
    A=fun_txt
    B=current_population
    number=0
    for i in range(len(B)):
        nn=0
        for j in range(len(A)):
            if(dominate(A[j], B[i])):
                nn=nn+1#B[i]被A支配的个体数目+1
        if (nn != 0 ):
            number=number+1
    C_AB=float(number/len(B))
    print("C_AB：%2f"%C_AB)
    #-----Distance from Representatives in the PF(D-metric)-----
    A=current_population
    P=fun_txt
    min_d=0
    for i in range(len(P)):
        temp=[]
        for j in range(len(A)):
            dd=0
            for k in range(F_NUM):
                dd=dd+float((P[i][k] - A[j].fitness[k]) ** 2)
            temp.append(math.sqrt(dd))
        min_d=min_d+np.min(temp)
    D_AP=float(min_d/len(P))
    print("D_AP：%2f"%D_AP)