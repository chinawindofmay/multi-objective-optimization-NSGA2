# -*- coding:utf-8 -*-

"""
用于初始化provider对象
"""

import math
import json
#ORACLE库
import cx_Oracle
from pymongo import MongoClient
import a_mongo_operater
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import seaborn as sns
import zlib
import matplotlib as mpl
# plt.rcParams['font.sans-serif']=['NSimSun'] # 用来正常显示中文标签
# matplotlib.rcParams['font.family']='SimHei' #黑体
plt.rcParams['font.sans-serif']=['SimHei']#黑体
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例
from matplotlib import cm



def show_3D_graph(objectives_fitness):  # 画图
    fig = plt.figure()
    ax = Axes3D(fig)
    font2 = {'family': 'SimHei',
             'weight': 'normal',
             'size': 15,
             }
    ax.set_xlabel("行驶时间", font2)
    ax.set_xlabel("覆盖人口反比", font2)
    ax.set_ylabel("不公平性", font2)
    ax.set_zlabel("建设成本", font2)
    # 删除重复行
    uniques = np.unique(objectives_fitness,axis = 0)
    # 按照第三列排序，自小往大排序，这样序号越小，建设成本越低，序号越大建设成本越高
    objectives_fitness=uniques[np.argsort(uniques[:, 2]), :]
    type1 = ax.scatter(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2],s=80, marker='+', c='r')
    # plt.legend((type1), (u'Non-dominated solution'))
    # 添加编号
    list_id=[str(i) for i in range(len(objectives_fitness[:, 0]))]
    # 添加标注
    for i in range(len(objectives_fitness[:, 0])):
        ax.text(objectives_fitness[i, 0], objectives_fitness[i, 1], objectives_fitness[i, 2], list_id[i], fontsize=12)
        # ax.annotate(list_id[i], xy=(objectives_fitness[i:, 0], objectives_fitness[i:, 1], objectives_fitness[i, 2]), xytext=(objectives_fitness[i:, 0], objectives_fitness[i:, 1], objectives_fitness[i, 2]))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标

    #构建网格趋势面
    n = 100
    u = np.linspace(0, 1, n)  # 创建一个等差数列
    X,Y = np.meshgrid(u, u)  # 转化成矩阵
    # ("./log/njall_solutions_nsga2_up200_time033_drivingtime.npz")
    # Z = +0.619 * X*X + 1.057 * X*Y + 1.188 * Y*Y - 1.882 * X - 2.181 * Y + 1.441
    # ("./log/njall_solutions_nsga2_up200_time033_coverpeople.npz")
    Z = +0.790 * X*X - 1.084 * X*Y  - 0.275 * Y*Y - 1.501 * X + 0.520 * Y + 0.653  #0.753
    # ("./log/njall_solutions_nsga2_up100_time0.16_poplimit_static.npz")
    # Z = +1.376 * X*X + 1.538 *  X*Y + 1.161 *Y*Y - 3.046 * X - 2.730 * Y + 1.798  #1.998  static
    # # ("./log/njall_solutions_nsga2_up100_time0.16_poplimit_dynamic.npz")
    # Z = +1.260 *  X*X + 2.306 *X*Y + 2.012 * Y*Y - 2.545 * X - 3.222 *  Y + 1.259 #1.459  dynamic
    surf = ax.plot_surface(X, Y, Z,
                           rstride=5,  # rstride（row）指定行的跨度
                           cstride=5,  # cstride(column)指定列的跨度
                           cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
    #图例
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def fun(x):  # 处理符号问题
    round(x, 2)
    if x >= 0:
        return '+' + str(x)
    else:
        return str(x)

# 主函数
def fit_3d_surface(X,Y,Z):
    fig = plt.figure() #建立一个新的图像
    # ax = Axes3D(fig) #建立一个三维坐标系
    # ax.scatter(X,Y,Z,c='b')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    n=len(X)
    # 求方程系数
    sigma_x = 0
    for i in X: sigma_x += i
    sigma_y = 0
    for i in Y: sigma_y += i
    sigma_z = 0
    for i in Z: sigma_z += i
    sigma_x2 = 0
    for i in X: sigma_x2 += i * i
    sigma_y2 = 0
    for i in Y: sigma_y2 += i * i
    sigma_x3 = 0
    for i in X: sigma_x3 += i * i * i
    sigma_y3 = 0
    for i in Y: sigma_y3 += i * i * i
    sigma_x4 = 0
    for i in X: sigma_x4 += i * i * i * i
    sigma_y4 = 0
    for i in Y: sigma_y4 += i * i * i * i
    sigma_x_y = 0
    for i in range(n):
        sigma_x_y += X[i] * Y[i]
    # print(sigma_xy)
    sigma_x_y2 = 0
    for i in range(n): sigma_x_y2 += X[i] * Y[i] * Y[i]
    sigma_x_y3 = 0
    for i in range(n): sigma_x_y3 += X[i] * Y[i] * Y[i] * Y[i]
    sigma_x2_y = 0
    for i in range(n): sigma_x2_y += X[i] * X[i] * Y[i]
    sigma_x2_y2 = 0
    for i in range(n): sigma_x2_y2 += X[i] * X[i] * Y[i] * Y[i]
    sigma_x3_y = 0
    for i in range(n): sigma_x3_y += X[i] * X[i] * X[i] * Y[i]
    sigma_z_x2 = 0
    for i in range(n): sigma_z_x2 += Z[i] * X[i] * X[i]
    sigma_z_y2 = 0
    for i in range(n): sigma_z_y2 += Z[i] * Y[i] * Y[i]
    sigma_z_x_y = 0
    for i in range(n): sigma_z_x_y += Z[i] * X[i] * Y[i]
    sigma_z_x = 0
    for i in range(n): sigma_z_x += Z[i] * X[i]
    sigma_z_y = 0
    for i in range(n): sigma_z_y += Z[i] * Y[i]
    # print("-----------------------")
    # 给出对应方程的矩阵形式
    a = np.array([[sigma_x4, sigma_x3_y, sigma_x2_y2, sigma_x3, sigma_x2_y, sigma_x2],
                  [sigma_x3_y, sigma_x2_y2, sigma_x_y3, sigma_x2_y, sigma_x_y2, sigma_x_y],
                  [sigma_x2_y2, sigma_x_y3, sigma_y4, sigma_x_y2, sigma_y3, sigma_y2],
                  [sigma_x3, sigma_x2_y, sigma_x_y2, sigma_x2, sigma_x_y, sigma_x],
                  [sigma_x2_y, sigma_x_y2, sigma_y3, sigma_x_y, sigma_y2, sigma_y],
                  [sigma_x2, sigma_x_y, sigma_y2, sigma_x, sigma_y, n]])
    b = np.array([sigma_z_x2, sigma_z_x_y, sigma_z_y2, sigma_z_x, sigma_z_y, sigma_z])
    # 高斯消元解线性方程
    res = np.linalg.solve(a, b)
    # print(a)
    # print(b)
    # print(x)
    # print("-----------------------")

    # 输出方程形式
    print("z=%.6s*x^2%.6s*xy%.6s*y^2%.6s*x%.6s*y%.6s" % (
    fun(res[0]), fun(res[1]), fun(res[2]), fun(res[3]), fun(res[4]), fun(res[5])))
    # 画曲面图和离散点
    fig = plt.figure()  # 建立一个空间
    ax = fig.add_subplot(111, projection='3d')  # 3D坐标
    n = 100
    u = np.linspace(0, 1, n)  # 创建一个等差数列
    x, y = np.meshgrid(u, u)  # 转化成矩阵
    # 给出方程
    z = res[0] * x * x + res[1] * x * y + res[2] * y * y + res[3] * x + res[4] * y + res[5]
    # 画出曲面
    ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap=cm.jet)
    # 画出点
    ax.scatter(X, Y, Z, c='r')
    plt.show()


if __name__=="__main__":
    D1_npzfile = np.load("./log/njall_solutions_nsga2_up200_time033_drivingtime.npz")
    D1_npzfile = np.load("./log/njall_solutions_nsga2_up200_time033_coverpeople.npz")
    # D1_npzfile = np.load("./log/njall_solutions_nsga2_up100_time0.16_poplimit_static.npz")
    # D1_npzfile = np.load("./log/njall_solutions_nsga2_up100_time0.16_poplimit_dynamic.npz")
    population_front0 = D1_npzfile["population_front0"]
    objectives_fitness = D1_npzfile["objectives_fitness"]
    # 拟合曲面方程
    fit_3d_surface(objectives_fitness[:, 0], objectives_fitness[:, 1], objectives_fitness[:, 2])
    #将结果评价值显示出来
    # show_3D_graph(objectives_fitness)