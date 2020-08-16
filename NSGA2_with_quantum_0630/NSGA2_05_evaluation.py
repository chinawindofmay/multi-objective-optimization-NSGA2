#coding:utf-8
# Program Name: Quantum NSGA-II
# 测试函数所在的位置
# https://blog.csdn.net/miscclp/article/details/38102831
# https://wenku.baidu.com/view/9dd19d7dcd84b9d528ea81c758f5f61fb73628ae.html
#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



####################################################BEGIN：工具函数################################################
#希望求最小值
#三维平面
def y1(x1,x2):
    value = -(200-3*x1-7*x2)
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



def ZDT1_f1(population):
    f1 = population[:, 0] #因为传入的是整个population
    return f1

def ZDT1_f2(population):
    sum1 = np.array([np.sum(population[i, 1:]) for i in range(len(population[:, 0]))])  # 因为传入的是整个population
    g = (1 + 9 * (sum1 / (30 - 1))).astype(np.float)
    f2 = g * (1 - np.power(population[:, 0] / g, 0.5))
    return f2


def ZDT2_f1(population):
    f1 = population[:, 0] #因为传入的是整个population
    return f1

def ZDT2_f2(population):
    sum1 = np.array([np.sum(population[i, 1:]) for i in range(len(population[:, 0]))])  # 因为传入的是整个population
    g = (1 + 9 * (sum1 / (30 - 1))).astype(np.float)
    f2 = g * (1 - np.power(population[:, 0] / g, 2))
    return f2

def ZDT3_f1(population):
    f1 = population[:, 0] #因为传入的是整个population
    return f1

def ZDT3_f2(population):
    sum1 = np.array([np.sum(population[i, 1:]) for i in range(len(population[:, 0]))]) #因为传入的是整个population
    g = (1 + 9 * sum1 / 29).astype(np.float)
    h = 1 - np.power(population[:, 0] / g, 0.5) - (population[:, 0] / g) * np.sin(10 * np.pi * population[:, 0])
    f2 = g * h
    return f2

def ZDT4_f1(population):
    f1 = population[:, 0] #因为传入的是整个population
    return f1

def ZDT4_f2(population):
    sum1 = np.array([np.sum(np.power(population[i,1:],2)-10*np.cos(4*np.pi*population[i,1:])) for i in range(len(population[:, 0]))]) #因为传入的是整个population
    g = (91+ sum1).astype(np.float)
    f2 = (g * (1-(population[:, 0]/g)**0.5)).astype(np.float)
    return f2

def ZDT6_f1(population):
    f1 = (1 - np.exp(-4 * population[:, 0]) * (np.sin(6 * np.pi * population[:, 0])) ** 6).astype(np.float) #因为传入的是整个population
    return f1

def ZDT6_f2(population):
    sum1 = np.array([np.sum(population[i,1:]) for i in range(len(population[:, 0]))]) #因为传入的是整个population
    g = (1+9*((sum1/(10-1))**0.25)).astype(np.float)
    f1 = (1 - np.exp(-4 * population[:, 0]) * (np.sin(6 * np.pi * population[:, 0])) ** 6).astype(np.float)
    f2 = g*(1-(f1/g)**2)
    return f2


def schaffer2_f1(population):
    f1=np.full(len(population[:,0]),0.0)
    for i in range(len(population[:, 0])):
        if population[i,0]<=1:
            f1[i]=-1*population[i,0]
        elif population[i,0]>1 and population[i,0]<=3:
            f1[i] = population[i, 0]-2
        elif population[i,0]>3 and population[i,0]<=4:
            f1[i] = 4-population[i, 0]
        elif population[i,0]>4:
            f1[i] = population[i, 0]-4
    return f1

def schaffer2_f2(population):
    f2=(np.power(population[:, 0]-5,2)).astype(np.float)
    return f2




def MODA_fitness_functions( chromosome, fitness_function_name, x_num):
    chromosome = chromosome
    if (fitness_function_name == 'ZDT1'):
        f1 = float(chromosome[0])
        sum1 = np.sum(chromosome[1:])
        g = float(1 + 9 * (sum1 / (x_num - 1)))
        f2 = g * (1 - (f1 / g) ** (0.5))
        f = [f1, f2]
    elif (fitness_function_name == 'ZDT2'):
        f1 = float(chromosome[0])
        sum1 = np.sum(chromosome[1:])
        g = float(1 + 9 * (sum1 / (x_num - 1)))
        f2 = g * (1 - (f1 / g) ** 2)
        f = [f1, f2]
    elif (fitness_function_name == 'ZDT3'):
        f1 = float(chromosome[0])
        sum1 = np.sum(chromosome[1:])
        g = float(1 + 9 * (sum1 / (x_num - 1)))
        h = 1 - (f1 / g) ** 0.5 - (f1 / g) * math.sin(10 * math.pi * f1)
        f2 = g * h
        f = [f1, f2]
    elif (fitness_function_name == 'ZDT4'):
        f1 = float(chromosome[0])
        sum2 = np.sum(np.power(chromosome[1:], 2) - 10 * np.cos(4 * np.pi * chromosome[1:]))
        g = float(91 + sum2)
        f2 = g * (1 - (f1 / g) ** (0.5))
        f = [f1, f2]
    elif (fitness_function_name == 'ZDT6'):
        f1 = float(1 - math.exp(-4 * chromosome[0]) * (math.sin(6 * math.pi * chromosome[0])) ** 6)
        sum1 = np.sum(chromosome[1:])
        g = 1 + 9 * ((sum1 / (x_num - 1)) ** 0.25)
        f2 = g * (1 - (f1 / g) ** 2)
        f = [f1, f2]
    elif (fitness_function_name == 'DTLZ1'):
        sum1 = np.sum(np.power(chromosome[2:] - 0.5, 2) - np.cos(20 * np.pi * (chromosome[2:] - 0.5)))
        g = float(100 * (x_num - 2) + 100 * sum1)
        f1 = float((1 + g) * chromosome[0] * chromosome[1])
        f2 = float((1 + g) * chromosome[0] * (1 - chromosome[1]))
        f3 = float((1 + g) * (1 - chromosome[0]))
        f = [f1, f2, f3]
    elif (fitness_function_name == 'DTLZ2'):
        g = np.sum(chromosome[2:] ** 2)
        f1 = float((1 + g) * math.cos(0.5 * math.pi * chromosome[0]) * math.cos(0.5 * math.pi * chromosome[1]))
        f2 = float((1 + g) * math.cos(0.5 * math.pi * chromosome[0]) * math.sin(0.5 * math.pi * chromosome[1]))
        f3 = float((1 + g) * math.sin(0.5 * math.pi * chromosome[0]))
        f = [f1, f2, f3]


#三维制图表达
def draw_3d_plot(MIN_X, MAX_X,population, y1_values,y2_values,y3_values,f1,f2,f3):
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
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f1(x1, x2), rstride=1, cstride=1, cmap=plt.cm.spring)

    # 曲面y2的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f2(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)

    # 曲面y3的
    x1 = np.arange(MIN_X, MAX_X+1, 1)
    x2 = np.arange(MIN_X, MAX_X+1, 1)
    x1, x2 = np.meshgrid(x1, x2)
    ax3d.plot_surface(x1, x2, f3(x1, x2), rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    plt.show()

#三维制图表达
def draw_3d_plot_test3333(MIN_X, MAX_X):
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

def draw_2d_plot_gd_and_sp(MAX_GEN, gd_array, sp_array):
    fig = plt.figure(figsize=(4, 8))
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122)
    x=np.arange(1,MAX_GEN+1,1)
    ax11.set_xlabel('interation num', fontsize=15)
    ax11.set_ylabel('gd', fontsize=15)
    ax11.plot(x, gd_array)
    ax11.set_xlabel('interation num', fontsize=15)
    ax11.set_ylabel('sp', fontsize=15)
    ax12.plot(x, sp_array)


def draw_2d_plot_gd_and_sp_compason(MAX_GEN,
                                        traditional_gd_array, traditional_sp_array,
                                        no_rotate_gd_array, no_rotate_sp_array,
                                        rotate_gd_array, rotate_sp_array):
    fig = plt.figure(figsize=(4, 8))
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122)
    x = np.arange(1, MAX_GEN + 1, 1)
    ax11.set_xlabel('interation num', fontsize=15)
    ax11.set_ylabel('general distance', fontsize=15)
    ax11.plot(x, traditional_gd_array,"g-",label='traditional NSGA2')
    ax11.plot(x, no_rotate_gd_array,"r-",label='quantum NSGA2 with no rotate')
    ax11.plot(x, rotate_gd_array,"b-",label='quantum NSGA2 with rotate')
    ax11.legend()
    ax12.set_xlabel('interation num', fontsize=15)
    ax12.set_ylabel('space presentation', fontsize=15)
    ax12.plot(x, traditional_sp_array,"g--",label='traditional NSGA2')
    ax12.plot(x, no_rotate_sp_array,"r--",label='quantum NSGA2 with no rotate')
    ax12.plot(x, rotate_sp_array,"b--",label='quantum NSGA2 with rotate')
    ax12.legend()
# 二维制图表达
def draw_2d_plot_evaluation_pof_3(y1_values, y2_values, y3_values):
    fig = plt.figure(figsize=(4, 12))
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)
    ax11.set_xlabel('y1', fontsize=15)
    ax11.set_ylabel('y2', fontsize=15)
    ax11.scatter(y1_values, y2_values)
    ax12.set_xlabel('y2', fontsize=15)
    ax12.set_ylabel('y3', fontsize=15)
    ax12.scatter(y2_values, y3_values)
    ax13.set_xlabel('y1', fontsize=15)
    ax13.set_ylabel('y3', fontsize=15)
    ax13.scatter(y1_values, y3_values)
    plt.show()

# 二维制图表达
def draw_2d_plot_evaluation_pof_2(y1_values, y2_values):
    fig = plt.figure(figsize=(12, 12))
    ax11 = fig.add_subplot(111)
    ax11.set_xlabel('y1', fontsize=15)
    ax11.set_ylabel('y2', fontsize=15)
    ax11.scatter(y1_values, y2_values)
    plt.show()

def draw_2d_plot_evaluation_pof_compason(traditional_pof_y1_values_np, traditional_pof_y2_values_np, traditional_pof_y3_values_np,
                                                    no_rotate_pof_y1_values_np, no_rotate_pof_y2_values_np,no_rotate_pof_y3_values_np,
                                                    rotate_pof_y1_values_np, rotate_pof_y2_values_np,rotate_pof_y3_values_np):
    fig = plt.figure(figsize=(12, 12))
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)
    ax21 = fig.add_subplot(231)
    ax22 = fig.add_subplot(232)
    ax23 = fig.add_subplot(233)
    ax31 = fig.add_subplot(331)
    ax32 = fig.add_subplot(332)
    ax33 = fig.add_subplot(333)

    ax11.set_xlabel('y1', fontsize=15)
    ax11.set_ylabel('y2', fontsize=15)
    ax11.scatter(traditional_pof_y1_values_np, traditional_pof_y2_values_np,c="g",label='traditional NSGA2')
    ax12.set_xlabel('y2', fontsize=15)
    ax12.set_ylabel('y3', fontsize=15)
    ax12.scatter(traditional_pof_y2_values_np, traditional_pof_y3_values_np,c="g",label='traditional NSGA2')
    ax13.set_xlabel('y1', fontsize=15)
    ax13.set_ylabel('y3', fontsize=15)
    ax13.scatter(traditional_pof_y1_values_np, traditional_pof_y3_values_np,c="g",label='traditional NSGA2')

    ax21.set_xlabel('y1', fontsize=15)
    ax21.set_ylabel('y2', fontsize=15)
    ax21.scatter(no_rotate_pof_y1_values_np, no_rotate_pof_y2_values_np,c="b",label='quantum NSGA2 with no rotate')
    ax22.set_xlabel('y2', fontsize=15)
    ax22.set_ylabel('y3', fontsize=15)
    ax22.scatter(no_rotate_pof_y2_values_np, no_rotate_pof_y3_values_np,c="b",label='quantum NSGA2 with no rotate')
    ax23.set_xlabel('y1', fontsize=15)
    ax23.set_ylabel('y3', fontsize=15)
    ax23.scatter(no_rotate_pof_y1_values_np, no_rotate_pof_y3_values_np,c="b",label='quantum NSGA2 with no rotate')

    ax31.set_xlabel('y1', fontsize=15)
    ax31.set_ylabel('y2', fontsize=15)
    ax31.scatter(rotate_pof_y1_values_np, rotate_pof_y2_values_np,c="y",label='quantum NSGA2 with rotate')
    ax32.set_xlabel('y2', fontsize=15)
    ax32.set_ylabel('y3', fontsize=15)
    ax32.scatter(rotate_pof_y2_values_np, rotate_pof_y3_values_np,c="y",label='quantum NSGA2 with rotate')
    ax33.set_xlabel('y1', fontsize=15)
    ax33.set_ylabel('y3', fontsize=15)
    ax33.scatter(rotate_pof_y1_values_np, rotate_pof_y3_values_np,c="y",label='quantum NSGA2 with rotate')
    fig.legend()
    plt.show()
####################################################END:工具函数#####################################





