# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:34:18 2018
@author: lj
参考：https://github.com/shiluqiang/QGA_python
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
####################################################################BEGIN 完成类定义######################################################
## 2. QGA算法
class QGA(object):
    ###2.1 类初始化
    '''定义QGA类
    '''
    def __init__(self, population_size, X_num, chromosome_length, max_value, min_value, iter_num, deta, F):
        '''初始化类参数
        population_size(int):种群数
        chromosome_num(int):染色体数，对应需要寻优的参数个数
        chromosome_length(int):染色体长度
        max_value(float):染色体十进制数值最大值
        min_value(float):染色体十进制数值最小值
        iter_num(int):迭代次数
        deta(float):量子旋转角度
        '''
        self.population_size = population_size
        self.X_num = X_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.min_value = min_value
        self.iter_num = iter_num
        self.deta = deta
        self.F=F

    ### 2.2 种群的量子形式初始化
    def initial_population_angle_np(self):
        '''种群初始化
        input:self(object):QGA类
        output:population_Angle(list):种群的量子角度列表
               population_Angle2(list):空的种群的量子角度列表，用于存储交叉后的量子角度列表
        '''
        initial_population_angle_np=np.random.random(size=(self.X_num, self.population_size, self.chromosome_length))
        initial_population_angle_np= np.pi * 2 *initial_population_angle_np
        return initial_population_angle_np

    def population_qubit_np(self, population_angle):
        '''将初始化的量子角度序列转换为种群的量子系数列表
        input:self(object):QGA类
              population_Angle(list):种群的量子角度列表
        output:population_Q(list):种群的量子系数列表
        '''
        population_qubit_np = np.full((population_angle.shape[0], population_angle.shape[1], 2, population_angle.shape[2]), 0.0)
        for i in range(len(population_angle)):  #2
            tmp1 = []  ##存储每个染色体所有取值的量子系数对
            for j in range(len(population_angle[i])):#200
                tmp_pair = []  ## 存储每个染色体的每个取值的量子对 2
                tmp3 = []  ## 存储量子对的一半,20
                tmp4 = []  ## 存储量子对的另一半
                for m in range(len(population_angle[i][j])):#20
                    a = population_angle[i][j][m]
                    tmp3.append(np.sin(a))
                    tmp4.append(np.cos(a))
                tmp_pair.append(tmp3)
                tmp_pair.append(tmp4)
                tmp1.append(tmp_pair)
            population_qubit_np[i]=tmp1
        # print(np.array(population_qubit_np).shape)  # 2 200 2 20
        return population_qubit_np

    ### 2.3 计算适应度函数值
    def translation_bit_np(self, population_Q):
        '''将种群的量子列表转换为二进制列表
        input:self(object):QGA类
              population_Q(list):种群的量子列表
        output:population_Binary:种群的二进制列表
        '''
        initial_judge = np.random.random(size=(self.X_num, self.population_size, self.chromosome_length))
        population_bit_np=np.array(population_Q[:,:,0,:]>initial_judge,np.int)
        # print(population_bit_np.shape)
        return population_bit_np


    def fitness_np(self, population_bit):
        '''求适应度函数数值列表，本实验采用的适应度函数为RBF_SVM的3_fold交叉验证平均值
        input:self(object):QGA类
              population_Binary(list):种群的二进制列表
        output:fitness_value(list):适应度函数值类表
               parameters(list):对应寻优参数的列表
        '''
        ##1.染色体的二进制表现形式转换为十进制并设置在[min_value,max_value]之间
        X = np.full((self.X_num, self.population_size), 0.0)  ##存储所有参数的可能取值
        for i in range(len(population_bit)):
            tmp1 = []  ##存储一个参数的可能取值
            for j in range(len(population_bit[i])):
                ##计算二进制对应的十进制数值
                #int("110100010",2) = 418
                # for m in range(len(population_bit[i][j])):
                #     total += population_bit[i][j][m] * math.pow(2, m)
                total=int("".join('%s' %id for id in list(population_bit[i][j])),2)
                value = (total * (self.max_value - self.min_value)) / math.pow(2, len(population_bit[i][j])) + self.min_value  ##将十进制数值坐落在[min_value,max_value]之间
                tmp1.append(value)
            X[i]=tmp1
        #计算适应度函数
        fitness_value = np.array([self.F(X[0][i],X[1][i]) for i in range(len(X[0]))])
        ##3.找到最优的适应度函数值和对应的参数二进制表现形式
        best_fitness = np.max(fitness_value)
        index=np.where(fitness_value==best_fitness)
        best_x = X[:,index[0][0]]
        best_x_bit = population_bit[:, index[0][0]]
        return X, fitness_value, best_x_bit, best_fitness, best_x

    ### 2.4 全干扰交叉
    def crossover_np(self, population_angle_np):
        '''对种群量子角度列表进行全干扰交叉
        input:self(object):QGA类
              population_Angle(list):种群的量子角度列表
        '''
        ## 初始化一个空列表，全干扰交叉后的量子角度列表
        population_Angle_crossover = np.full((self.X_num, self.population_size, self.chromosome_length), 0.0)
        for i in range(len(population_angle_np)):
            for j in range(len(population_angle_np[i])):
                for m in range(len(population_angle_np[i][j])):
                    ni = (j - m) % len(population_angle_np[i])
                    population_Angle_crossover[i][j][m] = population_angle_np[i][ni][m]
        return population_Angle_crossover

    ### 2.4 变异
    def mutation_np(self, population_angle_crossover_np, population_angle_np, current_x_bit, current_fitness):
        '''采用量子门变换矩阵进行量子变异
        input:self(object):QGA类
              population_Angle_crossover(list):全干扰交叉后的量子角度列表
        output:population_Angle_mutation(list):变异后的量子角度列表
        '''
        ##1.求出交叉后的适应度函数值列表
        population_qubit_np_crossover = self.population_qubit_np(population_angle_crossover_np)  ## 交叉后的种群量子系数列表
        population_bit_np_crossover = self.translation_bit_np(population_qubit_np_crossover)  ## 交叉后的种群二进制数列表
        X, fitnesses_crossover, best_x_bit_crossover, best_fitness_crossover, best_x = self.fitness_np(population_bit_np_crossover)  ## 交叉后的适应度函数值列表
        ##2.初始化每一个量子位的旋转角度
        rotation_angle_np = np.full((population_angle_crossover_np.shape[0], population_angle_crossover_np.shape[1], population_angle_crossover_np.shape[2]), 0.0)
        deta = self.deta
        ##3. 求每个量子位的旋转角度
        for i in range(self.X_num):
            for j in range(self.population_size):
                if fitnesses_crossover[j] <= current_fitness:
                    for m in range(self.chromosome_length):
                        s1 = 0
                        a1 = population_qubit_np_crossover[i][j][0][m]
                        b1 = population_qubit_np_crossover[i][j][1][m]
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 0 and a1 * b1 == 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a1 * b1 == 0:
                            s1 = -1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        rotation_angle_np[i][j][m] = deta * s1
                else:
                    for m in range(self.chromosome_length):
                        s2 = 0
                        a2 = population_qubit_np_crossover[i][j][0][m]
                        b2 = population_qubit_np_crossover[i][j][1][m]
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 0 and a2 * b2 > 0:
                            s2 = -1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 0 and a2 * b2 < 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][ m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a2 * b2 > 0:
                            s2 = -1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a2 * b2 < 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 0 and current_x_bit[i][m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a2 * b2 > 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a2 * b2 < 0:
                            s2 = -1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a2 * b2 > 0:
                            s2 = 1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a2 * b2 < 0:
                            s2 = -1
                        if population_bit_np_crossover[i][j][m] == 1 and current_x_bit[i][m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        rotation_angle_np[i][j][m] = deta * s2
        ###4. 根据每个量子位的旋转角度，生成种群新的量子角度列表
        population_angle_mutation_np = population_angle_crossover_np + rotation_angle_np
        ##5.变异后的适应度值求解
        population_qubit_np_mutation = self.population_qubit_np(population_angle_mutation_np)
        population_bit_np_mutation = self.translation_bit_np(population_qubit_np_mutation)
        X_mutation, fitnesses_mutation, best_x_bit_mutation, best_fitness_mutation, best_x_mutation = self.fitness_np(population_bit_np_mutation)
        return population_angle_mutation_np,fitnesses_crossover,fitnesses_mutation

    #精英策略
    def select_np(self,population_angle_current_np,population_angle_crossover_np, population_angle_mutation_np,fitnesses_current,fitnesses_crossover, fitnesses_mutation):
        angle_np=np.hstack((population_angle_current_np,population_angle_crossover_np, population_angle_mutation_np))
        fitnesses_np=np.hstack((fitnesses_current,fitnesses_crossover, fitnesses_mutation))
        #归一化
        fitnesses_np_standation=(fitnesses_np - np.min(fitnesses_np))/(np.max(fitnesses_np)-np.min(fitnesses_np))
        angle_selected_np = np.zeros(population_angle_current_np.shape)
        fitnesses_accumulate_probility=np.add.accumulate(fitnesses_np_standation)/np.sum(fitnesses_np_standation)
        randoms=np.random.rand(len(population_angle_current_np[0]))
        for i in range(len(randoms)):
            for j in range(len(fitnesses_accumulate_probility)-1):
                if fitnesses_accumulate_probility[j]<randoms[i] and fitnesses_accumulate_probility[j+1]>randoms[i]:
                    angle_selected_np[:,i,:]=angle_np[:,j+1,:]
                    break
                else:
                    continue
        return angle_selected_np




    ### 2.6 主函数
    def main(self):
        best_fitness_results = []
        mean_fitness_results = []
        median_fitness_results = []
        all_record_fitnesses = []
        best_fitness = -1000000
        best_x = []
        ## 种群初始化
        population_angle_current_np = self.initial_population_angle_np()
        ## 迭代
        for i in range(self.iter_num):
            #将角度转为量子位，一个X（x1,x2）解对应2*20个量子位，实际上只是用到了1*20个的量子位，sin(a)那部分
            population_qubit_np = self.population_qubit_np(population_angle_current_np)
            ## 将量子位转换为二进制形式，量子bit决定了编码的多样性，这是非常关键的。
            population_bit_np = self.translation_bit_np(population_qubit_np)
            ## 计算本次迭代的适应度函数值列表，最优适应度函数值及对应的参数
            X, fitnesses_current, current_x_bit, iteration_best_fitness, current_x = self.fitness_np(population_bit_np)
            ## 找出到目前为止最优的适应度函数值和对应的参数
            if iteration_best_fitness > best_fitness:
                best_fitness = iteration_best_fitness
                best_x = current_x
            print('iteration is :', i + 1, ';Best x:', best_x, ';Best fitness', best_fitness)
            best_fitness_results.append(best_fitness)
            mean_fitness_results.append(np.mean(fitnesses_current))
            median_fitness_results.append(np.median(fitnesses_current))
            all_record_fitnesses.append(fitnesses_current)
            ## 全干扰交叉，存在问题
            population_angle_crossover_np = self.crossover_np(population_angle_current_np)
            ## 量子旋转变异
            population_angle_mutation_np,fitnesses_crossover,fitnesses_mutation = self.mutation_np(population_angle_crossover_np, population_angle_current_np, current_x_bit, iteration_best_fitness)
            ## 增加精英机制
            population_angle_current_np=self.select_np(population_angle_current_np,population_angle_crossover_np, population_angle_mutation_np, fitnesses_current,fitnesses_crossover, fitnesses_mutation)
        ## 结果展示
        # best_fitness_results.sort()
        return best_fitness_results,mean_fitness_results,median_fitness_results,all_record_fitnesses

##############################################################################END 完成类定义######################################################

#####################################################################BENGIN 工具函数###############################################################

# 绘制3D图形
def plot_3d(F):
    # 绘制函数三维图像
    fig = plt.figure()
    ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    X = np.linspace(*[MIN_VALUE,MAX_VALUE], 100)  # 等差数列
    Y = np.linspace(*[MIN_VALUE,MAX_VALUE], 100)
    X, Y = np.meshgrid(X, Y)  # 坐标矩阵，每个交叉点对应网格
    Z = F(X, Y)
    # 绘制3D图形，X,Y,Z:2D数组形式的数据值
    # rstride:数组行距（步长大小）
    # cstride:数组列距(步长大小）
    # cmap:曲面块颜色映射
    # color:曲面块颜色
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    # ax.set_zlim(-100, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.pause(3)  # 不仅可以绘图而且鼠标可以继续交互
    plt.show()


### 2.5 画出适应度函数值变化图
def plot_all_records(mean_fitness_results,median_fitness_results,all_record_fitnesses):
    '''画图
    '''
    X = [i+1 for i in range(ITER_NUM)]
    mean_tend = np.polyfit(X, mean_fitness_results, 2)
    p=np.poly1d(mean_tend)
    # plt.plot(X, best_fitness_results, 'r-', label='best')
    plt.plot(X, mean_fitness_results, 'b--', label="mean")
    plt.plot(X, median_fitness_results, 'y--', label="median")
    plt.plot(X, p(X), 'r-', label="median")
    plt.plot(all_record_fitnesses, 'g.')  # 每次循环的目标函数值
    plt.xlabel('Number of iteration', size=15)
    plt.ylabel('Value', size=15)
    plt.title('QGA')
    plt.legend()
    plt.show()

### 2.5 画出适应度函数值变化图
def plot_lines(best_fitness_results,mean_fitness_results,median_fitness_results,symbol_str,title,show_able):
    '''画图
    '''
    X = [i+1 for i in range(ITER_NUM)]
    plt.plot(X, best_fitness_results, 'r'+symbol_str, label=title+' best')
    plt.plot(X, mean_fitness_results, 'b'+symbol_str, label=title+" mean")
    plt.plot(X, median_fitness_results, 'y'+symbol_str, label=title+" median")
    plt.xlabel('Number of iteration', size=15)
    plt.ylabel('Value', size=15)
    plt.title(title)
    plt.legend()
    if show_able:
        plt.show()

def F1(x,y):
    return (200 - (x - 5) ** 2- (y - 6) ** 2)

def F2(x,y):
    # 计算适应度函数2 参考https://www.cnblogs.com/devilmaycry812839668/p/6371704.html?utm_source=itdadao&utm_medium=referral
    return (-(20 + x ** 2 + y ** 2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))))

def F3(x,y):
    # 计算适应度函数3 参考https://www.cnblogs.com/devilmaycry812839668/p/6371704.html?utm_source=itdadao&utm_medium=referral
    return -(0.5 + ((np.sin(x ** 2 - y ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (x ** 2 + y ** 2)) ** 2))
def F4(x,y):
    return (100*np.sqrt(np.abs(x-0.01*(y**2)))+0.01*np.abs(y+10))
#####################################################################END 工具函数###############################################################


POP_SIZE = 200
X_NUM = 2
CHROMOSOME_LENGTH = 20
MAX_VALUE = 5
MIN_VALUE = -5
ITER_NUM = 50
DETA = 0.05 * np.pi

if __name__ == '__main__':
    # plot_3d(F2)
    qga = QGA(POP_SIZE, X_NUM, CHROMOSOME_LENGTH, MAX_VALUE, MIN_VALUE, ITER_NUM, DETA,F2)
    best_fitness_results, mean_fitness_results, median_fitness_results, all_record_fitnesses=qga.main()
    plot_lines(best_fitness_results, mean_fitness_results, median_fitness_results)


