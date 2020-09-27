# -*- coding: utf-8 -*-
"""
跟换数据结构
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import random
from scipy.special import comb
from itertools import combinations
import copy
import math
import a_mongo_operater

class Accessibility:
    def __init__(self, low,up,popsize,old_providers_count, petential_new_providers_count, THROD, BEITA):
        self.low=low
        self.up=up
        self.pop_size=popsize
        self.old_providers_count=old_providers_count   #已建设施数量
        self.petential_new_providers_count = petential_new_providers_count # 定义自变量个数，候选新建的充电站数量
        self.THROD=THROD
        self.BEITA=BEITA

    ##############################################begin:fitness#########################################################

    # fitness计算
    def fitness(self, demands_provider_np, demands_pdd_np, popu, actural_popsize):
        """
        全局的适应度函数
        :param demands_np:
        :param cipf_np:
        :param popu:
        :return:
        """
        y1_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与居民区可达性
        y2_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 与居民区公平性
        y5_values_double = np.full(shape=(actural_popsize,),fill_value=1, dtype=np.float32)  # 覆盖人口

        for i in range(actural_popsize):
            solution=popu[i,:]
            solution_join = np.hstack((np.full(shape=self.old_providers_count, fill_value=0.01), solution))  #把前面的已建部分的0补齐；
            # 计算vj
            self.update_provider_vj(demands_provider_np, demands_pdd_np, solution_join)
            # start_time = time.time()
            self.calculate_single_provider_gravity_value_np(demands_provider_np,demands_pdd_np,solution_join)
            # 居民区 可达性适应度值，该值越大越好
            y1_values_double[i] = self.calculate_global_accessibility_numpy(demands_pdd_np)
            # 居民区  计算公平性数值，该值越小表示越公平
            y2_values_double[i] = self.calculate_global_equality_numpy(demands_pdd_np)
            # 覆盖人口，越大越好
            y5_values_double[i] = self.calcuate_global_cover_people(demands_provider_np)
        # 统一转成最小化问题
        pop_fitness = np.vstack((10/y1_values_double, y2_values_double, 10000000/y5_values_double)).T
        return pop_fitness

    def calculate_global_accessibility_numpy(self,demands_pdd_np):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        return np.nansum(demands_pdd_np[:,1])

    def calculate_global_equality_numpy(self, demands_pdd_np):
        """
        计算全局公平性值
        :param demands_numpy:
        :return:
        """
        # 计算sum population count，总的需求量
        sum_population_count = np.sum(demands_pdd_np[:, 0])
        a = np.nansum(
            np.multiply(
                np.divide(
                    demands_pdd_np[:, 0], sum_population_count
                ),
                demands_pdd_np[:,  1]
            )
        )
        # 计算方差
        var = np.nansum(
            np.power(
                demands_pdd_np[:, 1] - a,
                2
            )
        )
        return var

    def calcuate_global_cover_people(self, demands_provider_np):
        """
        获取覆盖全局人口的总和
        :param demands_np:
        :return:
        """
        vj=demands_provider_np[:, 0, 2]
        vj[np.isinf(vj)] = 0
        vj[np.isnan(vj)] = 0
        return np.nansum(vj)  # 获取到每一个的provider的vj值，然后求和，之所以用nan，是为了过滤掉nan的值

    def create_providers_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        构建以providers为主导的np数组
        :param demands_np:
        :param new_providers_count:
        :param solution_join:
        :return:
        """
        providers_np = np.full((demands_provider_np.shape[0], demands_provider_np.shape[1], 2), 0.000000001)
        providers_np[:, :, 0] = np.tile(demands_pdd_np[:,0],(demands_provider_np.shape[0],1)) #pdd
        providers_np[:, :, 1] = demands_provider_np[:,:,1]    #D_T
        # 将solution部分增加进去
        mask = solution_join==0
        mask=mask.reshape(mask.shape[0],1)
        mask2=np.tile(mask,(1,demands_provider_np.shape[1]))
        providers_np[:, :, 0][mask2]=0
        return providers_np

    def calculate_single_provider_gravity_value_np(self, demands_provider_np,demands_pdd_np,solution_join):
        """
        函数：计算重力值
        最费时的运算步骤，预计每一个solution，需要耗时0.5S
        :param demands_np:
        :param DEMANDS_COUNT:
        :return:
        """
        DEMANDS_COUNT=demands_provider_np.shape[1]
        # 执行求取每个需求点的重力值
        for i in range(DEMANDS_COUNT):
            mask = demands_provider_np[ :,i, 1] <= self.THROD  # D_T在一定有效范围内的
            gravity_value = np.nansum(
                np.divide(
                    np.add(demands_provider_np[ :, i,0][mask], solution_join[mask]),  # 快充+solution
                    np.multiply(
                        np.power(demands_provider_np[ :, i,1][mask], self.BEITA),  # D_T
                        demands_provider_np[ :,i, 2][mask]  # vj
                    )
                )
            )
            demands_pdd_np[i,1] = gravity_value

    def update_provider_vj(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        计算vj
        :param demands_np:
        :param DEMANDS_COUNT:
        :param PROVIDERS_COUNT:
        :return:
        """
        # 执行转置，计算得到每个provider的VJ
        # 以provider为一级主线，每个provider中存入所有社区的信息，以便用于vj的计算
        # 增加一个solution的过滤，因为新增部分有些建，有些不建
        effective_providers_np = self.create_providers_np(demands_provider_np, demands_pdd_np, solution_join)
        vj_list = []
        for j in range(effective_providers_np.shape[0]):
            mask = effective_providers_np[j, :, 1] <= self.THROD   #对交通时间做对比
            vj_list.append(
                np.sum(
                    np.divide(
                        effective_providers_np[j, :, 0][mask]
                        , np.power(effective_providers_np[j, :, 1][mask], self.BEITA)
                    )
                )
            )
        # 更新vj列
        vj_np=np.array(vj_list).reshape(len(vj_list),1)
        demands_provider_np[:,:,2]=np.tile(vj_np,(1,demands_provider_np.shape[1]))
        print("test")

    ##############################################end:fitness#########################################################


    def initial_population(self, popsize):
        """
        种群初始化
        :param popsize:
        :param ps:
        :return:
        """
        # 每一个供给者代表一个函数x
        # N为solution数量
        pop=np.full(shape=(popsize, self.petential_new_providers_count), fill_value=0.0)
        for m in range(popsize):
            # 方式一：新增方式，已建不调整；方式二：新建，已建可调整；方式三：总数约束，新建
            # 以下只针对未建服务设施的站点，部分新增设施，已建服务设施的站点，不做考虑
            creating_mark = np.random.randint(low=0, high=2,
                                              size=(self.petential_new_providers_count))  # 首先，区分已建和未建，针对未建的停车场生成0和1
            creating_ps = np.random.randint(low=self.low, high=self.up + 1,
                                            size=(self.petential_new_providers_count))  # 新建的设施
            pop[m,:] = creating_mark * creating_ps
        return pop

    # 主函数
    def excute_with_population(self, demands_provider_np, demands_pdd_np):
        popu = self.initial_population(self.pop_size)  # 生成初始种群，修改了ps
        popu_fitness = self.fitness(demands_provider_np, demands_pdd_np, popu, self.pop_size)  # 计算适应度函数值

    # 主函数
    def excute_without_population(self, demands_provider_np, demands_pdd_np,popu):
        return self.fitness(demands_provider_np, demands_pdd_np, popu, self.pop_size)  # 计算适应度函数值


def jianye_ps_accessibility_3values():
    # 区
    POP_SIZE2 = 200  # 种群大小
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps"  # COLLECTION 名称
    DEMANDS_COUNT = 184  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 49
    OLD_PROVIDERS_COUNT = 40
    BEITA = 0.8  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 1  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np = mongo_operater_obj.find_records_format_numpy_2(0, DEMANDS_COUNT,
                                                                                         OLD_PROVIDERS_COUNT + PENTENTIAL_NEW_PROVIDERS_COUNT)  # 必须要先创建索引，才可以执行
    acce = Accessibility(low2, up2, POP_SIZE2, OLD_PROVIDERS_COUNT, PENTENTIAL_NEW_PROVIDERS_COUNT, THROD, BEITA)
    acce.excute_with_population(demands_provider_np, demands_pdd_np)

def test_self_defined_map_accessibility_3values():
    POP_SIZE2 = 5  # 种群大小
    DEMANDS_COUNT = 4  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NEW_PROVIDERS_COUNT = 1
    OLD_PROVIDERS_COUNT = 3
    BEITA = 1  # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 4  # 有效距离或者时间的阈值
    low2 = 3  # 最低阈值，至少建设3个
    up2 = 10  # 最高阈值，最多建设10个
    demands_provider_np=[
                         [[5,1,0.01],
                          [5,3,0.01],
                          [5,2,0.01],
                          [5,1,0.01]],   #p1

                        [[3,2,0.01],
                         [3,3,0.01],
                         [3,1,0.01],
                         [3,1,0.01]],   #p2

                        [[4,1,0.01],
                         [4,1,0.01],
                         [4,1,0.01],
                         [4,2,0.01]],   #p3

                        [[0,3,0.01],
                         [0,2,0.01],
                         [0,1,0.01],
                         [0,2,0.01]]]
    demands_pdd_np=[[10,0.0001],
                    [10,0.0001],
                    [15,0.0001],
                    [20,0.0001]]
    demands_provider_np=np.array(demands_provider_np)
    demands_pdd_np=np.array(demands_pdd_np)
    # 执行NSGA3
    nsga3 = Accessibility(low2, up2, POP_SIZE2, OLD_PROVIDERS_COUNT, PENTENTIAL_NEW_PROVIDERS_COUNT, THROD, BEITA)
    nsga3.excute_with_population(demands_provider_np, demands_pdd_np)

if __name__=="__main__":
    jianye_ps_accessibility_3values()
    # test()