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


## 2. QGA算法
class QGA(object):
    ###2.1 类初始化
    '''定义QGA类
    '''
    def __init__(self, population_size, qubit_num, chromosome_length, max_value, min_value, iter_num, deta):
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
        self.qubit_num = qubit_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.min_value = min_value
        self.iter_num = iter_num
        self.deta = deta

    ### 2.2 种群的量子形式初始化
    def initial_population_angle(self):
        '''种群初始化
        input:self(object):QGA类
        output:population_Angle(list):种群的量子角度列表
               population_Angle2(list):空的种群的量子角度列表，用于存储交叉后的量子角度列表
        '''
        initial_population = []
        for i in range(self.qubit_num):
            tmp1 = []  ##存储每个染色体所有取值的量子角度
            for j in range(self.population_size):
                tmp2 = []  ## 存储量子角度
                for m in range(self.chromosome_length):
                    a = np.pi * 2 * random.random()
                    tmp2.append(a)
                tmp1.append(tmp2)
            initial_population.append(tmp1)
        return initial_population

    def population_Q(self, population_Angle):
        '''将初始化的量子角度序列转换为种群的量子系数列表
        input:self(object):QGA类
              population_Angle(list):种群的量子角度列表
        output:population_Q(list):种群的量子系数列表
        '''
        population_Q = []

        for i in range(len(population_Angle)):
            tmp1 = []  ##存储每个染色体所有取值的量子系数对
            for j in range(len(population_Angle[i])):
                tmp2 = []  ## 存储每个染色体的每个取值的量子对
                tmp3 = []  ## 存储量子对的一半
                tmp4 = []  ## 存储量子对的另一半
                for m in range(len(population_Angle[i][j])):
                    a = population_Angle[i][j][m]
                    tmp3.append(np.sin(a))
                    tmp4.append(np.cos(a))
                tmp2.append(tmp3)
                tmp2.append(tmp4)
                tmp1.append(tmp2)
            population_Q.append(tmp1)
        return population_Q

    ### 2.3 计算适应度函数值
    def translation(self, population_Q):
        '''将种群的量子列表转换为二进制列表
        input:self(object):QGA类
              population_Q(list):种群的量子列表
        output:population_Binary:种群的二进制列表
        '''
        population_Binary = []
        for i in range(len(population_Q)):
            tmp1 = []  # 存储每个染色体所有取值的二进制形式
            for j in range(len(population_Q[i])):
                tmp2 = []  ##存储每个染色体每个取值的二进制形式
                for l in range(len(population_Q[i][j][0])):
                    if np.square(population_Q[i][j][0][l]) > random.random():
                        tmp2.append(1)
                    else:
                        tmp2.append(0)
                tmp1.append(tmp2)
            population_Binary.append(tmp1)
        return population_Binary

    def function(self, X):
        Z = []
        for i in range(len(X[0])):
            x = X[0][i]
            y = X[1][i]
            z = 1 - (x - 5) ** 2 - (y - 13) ** 2
            Z.append(z)
        return Z

    def fitness(self, population_Binary):
        '''求适应度函数数值列表，本实验采用的适应度函数为RBF_SVM的3_fold交叉验证平均值
        input:self(object):QGA类
              population_Binary(list):种群的二进制列表
        output:fitness_value(list):适应度函数值类表
               parameters(list):对应寻优参数的列表
        '''
        ##1.染色体的二进制表现形式转换为十进制并设置在[min_value,max_value]之间
        parameters = []  ##存储所有参数的可能取值
        for i in range(len(population_Binary)):
            tmp1 = []  ##存储一个参数的可能取值
            for j in range(len(population_Binary[i])):
                total = 0.0
                for l in range(len(population_Binary[i][j])):
                    total += population_Binary[i][j][l] * math.pow(2, l)  ##计算二进制对应的十进制数值
                value = (total * (self.max_value - self.min_value)) / math.pow(2, len(population_Binary[i][j])) + self.min_value  ##将十进制数值坐落在[min_value,max_value]之间
                tmp1.append(value)
            parameters.append(tmp1)
        fitness_value = self.function(parameters)

        ##2.适应度函数为RBF_SVM的3_fold交叉校验平均值
        # fitness_value = []
        # for l in range(len(parameters[0])):
        #     rbf_svm = svm.SVC(kernel = 'rbf', C = parameters[0][l], gamma = parameters[1][l])
        #     cv_scores = cross_validation.cross_val_score(rbf_svm,trainX,trainY,cv =3,scoring = 'accuracy')
        #     fitness_value.append(cv_scores.mean())

        ##3.找到最优的适应度函数值和对应的参数二进制表现形式
        best_fitness = -1 * (10 ** 20)
        mean_fitness=np.mean(fitness_value)
        all_record_fitness=fitness_value
        best_parameter = []
        best_parameter_Binary = []
        for j in range(len(population_Binary)):
            tmp2 = []
            best_parameter_Binary.append(tmp2)
            best_parameter.append(tmp2)

        for i in range(len(population_Binary[0])):
            if best_fitness < fitness_value[i]:
                best_fitness = fitness_value[i]
                for j in range(len(population_Binary)):
                    best_parameter_Binary[j] = population_Binary[j][i]
                    best_parameter[j] = parameters[j][i]
        return parameters, fitness_value, best_parameter_Binary, best_fitness, mean_fitness,all_record_fitness,best_parameter

    ### 2.4 全干扰交叉
    def crossover(self, population_Angle):
        '''对种群量子角度列表进行全干扰交叉
        input:self(object):QGA类
              population_Angle(list):种群的量子角度列表
        '''
        ## 初始化一个空列表，全干扰交叉后的量子角度列表
        population_Angle_crossover = []
        for i in range(self.qubit_num):
            tmp11 = []
            for j in range(self.population_size):
                tmp21 = []
                for m in range(self.chromosome_length):
                    tmp21.append(0.0)
                tmp11.append(tmp21)
            population_Angle_crossover.append(tmp11)
        for i in range(len(population_Angle)):
            for j in range(len(population_Angle[i])):
                for m in range(len(population_Angle[i][j])):
                    ni = (j - m) % len(population_Angle[i])
                    population_Angle_crossover[i][j][m] = population_Angle[i][ni][m]
        return population_Angle_crossover

    ### 2.4 变异
    def mutation(self, population_Angle_crossover, population_Angle, best_parameter_Binary, best_fitness):
        '''采用量子门变换矩阵进行量子变异
        input:self(object):QGA类
              population_Angle_crossover(list):全干扰交叉后的量子角度列表
        output:population_Angle_mutation(list):变异后的量子角度列表
        '''
        ##1.求出交叉后的适应度函数值列表
        population_Q_crossover = self.population_Q(population_Angle_crossover)  ## 交叉后的种群量子系数列表
        population_Binary_crossover = self.translation(population_Q_crossover)  ## 交叉后的种群二进制数列表
        parameters, fitness_crossover, best_parameter_Binary_crossover, best_fitness_crossover,mean_fitness,all_record_fitness, best_parameter = self.fitness(population_Binary_crossover)  ## 交叉后的适应度函数值列表
        ##2.初始化每一个量子位的旋转角度
        Rotation_Angle = []
        for i in range(len(population_Angle_crossover)):
            tmp1 = []
            for j in range(len(population_Angle_crossover[i])):
                tmp2 = []
                for m in range(len(population_Angle_crossover[i][j])):
                    tmp2.append(0.0)
                tmp1.append(tmp2)
            Rotation_Angle.append(tmp1)
        deta = self.deta
        ##3. 求每个量子位的旋转角度
        for i in range(self.qubit_num):
            for j in range(self.population_size):
                if fitness_crossover[j] <= best_fitness:
                    for m in range(self.chromosome_length):
                        s1 = 0
                        a1 = population_Q_crossover[i][j][0][m]
                        b1 = population_Q_crossover[i][j][1][m]
                        NP_population_Binary_crossover = np.array(population_Binary_crossover)
                        NP_best_parameter_Binary = np.array(best_parameter_Binary)
                        # print(NP_population_Binary_crossover.shape,NP_best_parameter_Binary.shape)
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a1 * b1 == 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        Rotation_Angle[i][j][m] = deta * s1
                else:
                    for m in range(self.chromosome_length):
                        s2 = 0
                        a2 = population_Q_crossover[i][j][0][m]
                        b2 = population_Q_crossover[i][j][1][m]
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][
                            m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 1 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][
                            m] == 1 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        Rotation_Angle[i][j][m] = deta * s2
        ###4. 根据每个量子位的旋转角度，生成种群新的量子角度列表
        for i in range(self.qubit_num):
            for j in range(self.population_size):
                for m in range(self.chromosome_length):
                    population_Angle[i][j][m] = population_Angle[i][j][m] + Rotation_Angle[i][j][m]
        return population_Angle

    ### 2.5 画出适应度函数值变化图
    def plot(self, mean_fitness_results,all_record_fitnesses):
        '''画图
        '''
        X = [i + 1 for i in range(self.iter_num)]
        mean_tend = np.polyfit(X, mean_fitness_results, 2)
        p = np.poly1d(mean_tend)
        # plt.plot(X, best_fitness_results, 'r-', label='best')
        plt.plot(X, mean_fitness_results, 'b--', label="mean")
        plt.plot(X, p(X), 'r-', label="mean tedency")
        plt.plot(all_record_fitnesses, 'g.')  # 每次循环的目标函数值
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel('Value', size=15)
        plt.title('QGA')
        plt.legend()
        plt.show()

    ### 2.6 主函数
    def main(self):
        mean_fitness_results = []
        all_record_fitnesses=[]
        best_fitness = 0.0
        best_parameter = []
        ## 种群初始化
        population_Angle = self.initial_population_angle()
        ## 迭代
        for i in range(self.iter_num):
            population_Q = self.population_Q(population_Angle)
            ## 将量子系数转换为二进制形式
            population_Binary = self.translation(population_Q)
            ## 计算本次迭代的适应度函数值列表，最优适应度函数值及对应的参数
            parameters, fitness_value, current_parameter_Binary, current_fitness,mean_fitness,all_record_fitness, current_parameter = self.fitness(population_Binary)
            ## 找出到目前为止最优的适应度函数值和对应的参数
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameter = current_parameter
            print('iteration is :', i + 1, ';Best parameters:', best_parameter, ';Best fitness', best_fitness)
            mean_fitness_results.append(mean_fitness)
            all_record_fitnesses.append(all_record_fitness)
            ## 全干扰交叉
            population_Angle_crossover = self.crossover(population_Angle)
            ## 量子旋转变异
            population_Angle = self.mutation(population_Angle_crossover, population_Angle, current_parameter_Binary,current_fitness)
        ## 结果展示
        self.plot(mean_fitness_results,all_record_fitnesses)
        print('Final parameters are :', parameters[-1])


if __name__ == '__main__':
    print('----------------2.Parameter Seting------------')
    population_size = 200
    chromosome_num = 2
    chromosome_length = 20
    max_value = 15
    min_value = 0.01
    iter_num = 10
    deta = 0.1 * np.pi
    print('----------------3.QGA_RBF_SVM-----------------')
    qga = QGA(population_size, chromosome_num, chromosome_length, max_value, min_value, iter_num, deta)
    qga.main()


