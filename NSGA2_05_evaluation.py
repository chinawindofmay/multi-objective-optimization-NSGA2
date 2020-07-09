#coding:utf-8
# Program Name: Quantum NSGA-II
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm and Quantum optimizatin mechanism.
# Author: Xinxin Zhou
# Supervisor: Prof. Linwang Yuan

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np



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
def draw_2d_plot_evaluation_pof(y1_values, y2_values,y3_values):
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





