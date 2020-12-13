# -*- coding: utf-8 -*-
"""
程序功能：实现MOEAD算法，测试函数为ZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
说明：遗传算子为模拟二进制交叉和多项式变异
作者：(晓风)
email: 18821709267@163.com 
最初建立时间：2018.10.10
最近修改时间：2018.10.10
参考论文：
MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
Qingfu Zhang, Senior Member, IEEE, and Hui Li
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
start=time.time()#开始计时

def funfun(fitness_function_name):
    if fitness_function_name== 'ZDT1':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt1=np.loadtxt('ZDT1.txt')
        plt.scatter(zdt1[:,0],zdt1[:,1],marker='o',color='green',s=40)
        f_txt=zdt1
        plt.show()
    elif fitness_function_name== 'ZDT2':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array=np.zeros(x_num)#决策变量的最小值
        X_max_array=np.ones(x_num)#决策变量的最大值
        zdt2=np.loadtxt('ZDT2.txt')
        plt.scatter(zdt2[:,0],zdt2[:,1],marker='o',color='green',s=40)
        f_txt=zdt2
        plt.show()
    elif fitness_function_name== 'ZDT3':
        f_num=2;#目标函数个数
        x_num=30;#决策变量个数
        X_min_array = np.zeros(x_num)  # 决策变量的最小值
        X_max_array = np.ones(x_num)  # 决策变量的最大值
        zdt3=np.loadtxt('ZDT3.txt')
        plt.scatter(zdt3[:,0],zdt3[:,1],marker='o',color='green',s=40)
        f_txt=zdt3
        plt.show()
    elif fitness_function_name== 'ZDT4':
        f_num=2;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array=np.array([0,-5,-5,-5,-5,-5,-5,-5,-5,-5],dtype=float)#决策变量的最小值
        X_max_array=np.array([1,5,5,5,5,5,5,5,5,5],dtype=float)#决策变量的最大值
        zdt4=np.loadtxt('ZDT4.txt')
        plt.scatter(zdt4[:,0],zdt4[:,1],marker='o',color='green',s=40)
        f_txt=zdt4
        plt.show()
    elif fitness_function_name== 'ZDT6':
        f_num=2;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array = np.zeros(x_num)  # 决策变量的最小值
        X_max_array = np.ones(x_num)  # 决策变量的最大值
        zdt6=np.loadtxt('ZDT6.txt')
        plt.scatter(zdt6[:,0],zdt6[:,1],marker='o',color='green',s=40)
        f_txt=zdt6
        plt.show()
    elif fitness_function_name== 'DTLZ1':
        f_num=3;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array = np.zeros(x_num)  # 决策变量的最小值
        X_max_array = np.ones(x_num)  # 决策变量的最大值
        dtlz1=np.loadtxt('DTLZ1.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz1[:,0],dtlz1[:,1],dtlz1[:,2],c='g')
        f_txt=dtlz1
        fig.show()
    elif fitness_function_name== 'DTLZ2':
        f_num=3;#目标函数个数
        x_num=10;#决策变量个数
        X_min_array = np.zeros(x_num)  # 决策变量的最小值
        X_max_array = np.ones(x_num)  # 决策变量的最大值
        dtlz2=np.loadtxt('DTLZ2.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz2[:,0],dtlz2[:,1],dtlz2[:,2],c='g')
        f_txt=dtlz2
        fig.show()
    return f_num,x_num,X_min_array,X_max_array,f_txt



class Individual():
    def __init__(self, chromosome, fitness_function_name,x_num):
        self.x=chromosome
        if (fitness_function_name== 'ZDT1'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g=float(1+9*(sum1/(x_num-1)))
            f2=g*(1-(f1/g)**(0.5))
            self.f=[f1,f2]
        elif (fitness_function_name== 'ZDT2'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g=float(1+9*(sum1/(x_num-1)))
            f2=g*(1-(f1/g)**2)
            self.f=[f1,f2]
        elif (fitness_function_name== 'ZDT3'):
            f1=float(chromosome[0])
            sum1 = np.sum(chromosome[1:])
            g=float(1+9*(sum1/(x_num-1)))
            h=1-(f1/g)**0.5-(f1/g)*math.sin(10*math.pi*f1)
            f2=g*h
            print("f2",f2)
            self.f=[f1,f2]
        elif (fitness_function_name== 'ZDT4'):
            f1=float(chromosome[0])
            sum2=np.sum(np.power(chromosome[1:],2)-10*np.cos(4*np.pi*chromosome[1:]))
            g=float(91+sum2)
            f2=g*(1-(f1/g)**(0.5))
            self.f=[f1,f2]
        elif (fitness_function_name== 'ZDT6'):
            f1=float(1 - math.exp(-4 * chromosome[0]) * (math.sin(6 * math.pi * chromosome[0])) ** 6)
            sum1=np.sum(chromosome[1:])
            g=1+9*((sum1/(x_num-1))**0.25)
            f2=g*(1-(f1/g)**2)
            self.f=[f1,f2]
        elif (fitness_function_name== 'DTLZ1'):
            sum1=np.sum(np.power(chromosome[2:]-0.5,2)-np.cos(20*np.pi*(chromosome[2:]-0.5)))
            g=float(100*(x_num-2)+100*sum1)
            f1=float((1+g) * chromosome[0] * chromosome[1])
            f2=float((1+g) * chromosome[0] * (1 - chromosome[1]))
            f3=float((1+g) * (1 - chromosome[0]))
            self.f=[f1,f2,f3]
        elif (fitness_function_name== 'DTLZ2'):
            g=np.sum(chromosome[2:] ** 2)
            f1=float((1+g) * math.cos(0.5 * math.pi * chromosome[0]) * math.cos(0.5 * math.pi * chromosome[1]))
            f2=float((1+g) * math.cos(0.5 * math.pi * chromosome[0]) * math.sin(0.5 * math.pi * chromosome[1]))
            f3=float((1+g) * math.sin(0.5 * math.pi * chromosome[0]))
            self.f=[f1,f2,f3]


def initial_population(pop_size, fitness_function_name, x_num, X_min_array, X_max_array, f_num):
    lamda=[]
    population=[]
    #种群初始化以及产生lamda
    for i in range(pop_size):
        temp=[]
        chromosome= (X_max_array - X_min_array) * np.random.random_sample(x_num) + X_min_array
        # for j in range(x_num):
        #     gene= X_min_array[j] + (X_max_array[j] - X_min_array[ j]) * random.random()
        #     chromosome.append(gene)
        population.append(Individual(chromosome,fitness_function_name,x_num))
        if (f_num==2):
            temp.append(float(i) / pop_size)
            temp.append(1.0 - float(i) / pop_size)
        elif(f_num==3):
            temp.append(float(i) / pop_size)
            temp.append(1.0 - float(i) / pop_size)
            temp.append(1.0 - float(pop_size - i - 1) / pop_size)
        lamda.append(temp)
    return population,lamda

def look_neighbor(lamda,T):
    B=[]
    for i in range(len(lamda)):
        temp=[]
        for j in range(len(lamda)):
            distance=np.sqrt((lamda[i][0]-lamda[j][0])**2+(lamda[i][1]-lamda[j][1])**2)
            temp.append(distance)
        l=np.argsort(temp)
        B.append(l[:T])
    return B

def bestvalue(P):
    best=[]
    for i in range(len(P[0].fitness_fun_mate_infor)):
        best.append(P[0].fitness_fun_mate_infor[i])
    for i in range(1,len(P)):
        for j in range(len(P[i].fitness_fun_mate_infor)):
            if P[i].fitness_fun_mate_infor[j]<best[j]:
                best[j]=P[i].fitness_fun_mate_infor[j]
    return best

def cross_mutation(parent1,parent2,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun):
    #模拟二进制交叉和多项式变异
    ###模拟二进制交叉
    off1=parent1
    off2=parent2
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
            off11=float(0.5*((1-gama)*parent1.x[j]+(1+gama)*parent2.x[j]))
            off22=float(0.5*((1-gama)*parent1.x[j]+(1+gama)*parent2.x[j]))
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
        off1=Individual(off1x,fitness_function_name,x_num)
        off2=Individual(off2x,fitness_function_name,x_num)
    #多项式变异
    if (random.random()<pm):
        off1x=[]
        off2x=[]
        for j in range(x_num):
            u2=random.random()
            if (u2<0.5):
                delta=float((2*u2)**(1/(yita2+1))-1)
            else:
                delta=float(1-(2*(1-u2))**(1/(yita2+1)))
            off11=float(off1.x[j] + delta)
            off22=float(off2.x[j] + delta)
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
        off1=Individual(off1x,fitness_function_name,x_num)
        off2=Individual(off2x,fitness_function_name,x_num)
    return off1,off2
            
def dominate(y1,y2):
    less=0#y1的目标函数值小于y2个体的目标函数值数目
    equal=0#y1的目标函数值等于y2个体的目标函数值数目
    greater=0#y1的目标函数值大于y2个体的目标函数值数目
    for i in range(len(y1.fitness_fun_mate_infor)):
        if y1.fitness_fun_mate_infor[i]>y2.fitness_fun_mate_infor[i]:
            greater=greater+1
        elif y1.fitness_fun_mate_infor[i]==y2.fitness_fun_mate_infor[i]:
            equal=equal+1
        else:
            less=less+1
    if(greater==0 and equal!=len(y1.fitness_fun_mate_infor)):
        return True#y1支配y2返回正确
    elif(less==0 and equal!=len(y1.fitness_fun_mate_infor)):
        return False#y2支配y1返回false
    else:
        return None
#列表版求C_AB
def Dominate(y1,y2):
    less=0#y1的目标函数值小于y2个体的目标函数值数目
    equal=0#y1的目标函数值等于y2个体的目标函数值数目
    greater=0#y1的目标函数值大于y2个体的目标函数值数目
    for i in range(len(y1)):
        if y1[i]>y2.fitness_fun_mate_infor[i]:
            greater=greater+1
        elif y1[i]==y2.fitness_fun_mate_infor[i]:
            equal=equal+1
        else:
            less=less+1
    if(greater==0 and equal!=len(y1)):
        return True#y1支配y2返回正确
    elif(less==0 and equal!=len(y1)):
        return False#y2支配y1返回false
    else:
        return None

def Tchebycheff(x,lamb,z):
    temp=[]
    for i in range(len(x.fitness_fun_mate_infor)):
        temp.append(np.abs(x.fitness_fun_mate_infor[i] - z[i]) * lamb[i])
    return np.max(temp)

def ws(x,lamba):
    temp=0.0
    for i in range(len(x.fitness_fun_mate_infor)):
        temp=temp+float(x.fitness_fun_mate_infor[i] * lamda[i])
    return temp

if __name__=="__main__":
    #------------------------参数输入--------------------------
    POP_SIZE=300#种群规模
    T=20#领域规模
    fitness_function_name= 'ZDT6'#测试函数DTLZ2
    f_num, x_num, X_min_array, X_max_array, fun_txt=funfun(fitness_function_name)
    MAX_GEN=250#最大进化代数
    CROSSOVER_PROBIBILITY=1#交叉概率
    MUTATION_PROBABILITY= 1 / x_num#变异概率
    yita1=2#模拟二进制交叉参数2
    yita2=5#多项式变异参数5
    #------------------------初始条件--------------------------
    population, lamda=initial_population(POP_SIZE, fitness_function_name, x_num, X_min_array, X_max_array, f_num)#计算均匀分布的N个权向量以及初始钟群
    ##计算任意两个权重向量间的欧式距离，查找每个权向量最近的T个权重向量的索引
    B=look_neighbor(lamda,T)
    ##初始化z
    z=bestvalue(population)
    #------------------------迭代更新--------------------------
    gen=1
    while(gen<=MAX_GEN):
        for i in range(POP_SIZE):
            ##基因重组，从B(i)中随机选取两个序列k，l
            k=random.randint(0,T-1)
            l=random.randint(0,T-1)
            y1,y2=cross_mutation(population[B[i][k]], population[B[i][l]], f_num, x_num, X_min_array, X_max_array, CROSSOVER_PROBIBILITY, MUTATION_PROBABILITY, yita1, yita2, fitness_function_name)
            if dominate(y1,y2):#y1支配y2返回正确
                y=y1
            else:
                y=y2
            ##更新z
            for j in range(len(z)):
                if y.f[j]<z[j]:
                    z[j]=y.f[j]
            ##更新领域解
            for j in range(len(B[i])):
                gte_xi=Tchebycheff(population[B[i][j]], lamda[B[i][j]], z)
                gte_y=Tchebycheff(y, lamda[B[i][j]], z)
                #gte_xi=ws(p[B[i][j]], lamda[B[i][j]])
                #gte_y=ws(y, lamda[B[i][j]])
                if (gte_y<=gte_xi):
                    population[B[i][j]]=y

        if gen%10 == 0:
            print("%d gen has completed!\n"%gen)
        gen=gen+1;
    end=time.time()
    print("循环时间：%2f秒"%(end-start))
    #------------------------画图对比--------------------------
    x=[]
    y=[]
    z=[]
    if f_num==2:
        for i in range(len(population)):
            x.append(population[i].f[0])
            y.append(population[i].f[1])
        plt.scatter(x,y,marker='o',color='red',s=40)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.show()
    elif f_num==3:
        for i in range(len(population)):
            x.append(population[i].f[0])
            y.append(population[i].f[1])
            z.append(population[i].f[2])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x,y,z,c='r')
        plt.show()

    #--------------------Coverage(C-metric)---------------------
    A=fun_txt
    B=population
    number=0
    for i in range(len(B)):
        nn=0
        for j in range(len(A)):
            if(Dominate(A[j],B[i])):
                nn=nn+1#B[i]被A支配的个体数目+1
        if (nn != 0 ):
            number=number+1
    C_AB=float(number/len(B))
    print("C_AB：%2f"%C_AB)
    #-----Distance from Representatives in the PF(D-metric)-----
    A=population
    P=fun_txt
    min_d=0
    for i in range(len(P)):
        temp=[]
        for j in range(len(A)):
            dd=0
            for k in range(f_num):
                dd=dd+float((P[i][k]-A[j].f[k])**2)
            temp.append(math.sqrt(dd))
        min_d=min_d+np.min(temp)
    D_AP=float(min_d/len(P))
    print("D_AP：%2f"%D_AP)