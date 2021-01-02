
import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib
import a_mongo_operater_PS_JY_TIME
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid', context='notebook')
# sns.set(style="ticks", color_codes=True)
import copy
import cx_Oracle



class PS_STA_dynamic():
    def test_global_accessibility_numpy(self, demands_pdd_np):
        """
        计算全局可达性的总和
        :param demands_numpy:
        :return:
        """
        # 适应度值，该值越大越好
        # 空间可达性数值实则代表相应研究单元人均拥有的公共服务资源数量，如每人医院床位数、每人教师数等，是一个人均的概念，入错此时再将其乘以人口，则得到了provider的总量。
        # 如果对其直接加和，是没有实际意义的，因为是一个均值单位的概念。
        # 获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        # 求取其均值，如果均值上来了，我们认为也是符合的
        calc_test_1= np.nansum(demands_pdd_np[:, 0]*demands_pdd_np[:, 4])
        calc_test_2= np.nansum(demands_pdd_np[:, 0]*demands_pdd_np[:, 4])
        calc_test_3= np.nansum(demands_pdd_np[:, 0]*demands_pdd_np[:, 4])
        calc_test_4= np.nansum(demands_pdd_np[:, 0]*demands_pdd_np[:, 4])
        print("理论值：",self.E_OLD_SUM,"计算得值：",calc_test_1,calc_test_2,calc_test_3,calc_test_4)

    def calculate_inequality_np(self, demands_pdd_np):
        """
        计算全局公平性值  可达性差异最小化为目标，通过除以一下，得到：实现设施布局的公平最大化问题
        :param demands_numpy:
        :return:
        """
        # 不同时刻的总人口
        pop_08_sum = np.nansum(demands_pdd_np[:, 0])
        pop_13_sum = np.nansum(demands_pdd_np[:, 1])
        pop_18_sum = np.nansum(demands_pdd_np[:, 2])
        pop_22_sum = np.nansum(demands_pdd_np[:, 3])
        SUM_CHARGE=self.E_OLD_SUM
        # 可达性差异==不公平性最小化 最小化为目标
        inequlity=np.full(shape=(demands_pdd_np.shape[0],4),fill_value=0.0)
        inequlity[:,0]= (demands_pdd_np[:, 4] - SUM_CHARGE / pop_08_sum) ** 2
        inequlity[:,1]= (demands_pdd_np[:, 5] - SUM_CHARGE / pop_13_sum) ** 2
        inequlity[:,2]= (demands_pdd_np[:, 6] - SUM_CHARGE / pop_18_sum) ** 2
        inequlity[:,3]= (demands_pdd_np[:, 7] - SUM_CHARGE / pop_22_sum) ** 2
        return inequlity

    def calcuate_global_cover_people(self, demands_provider_np,solution_join):
        """
        获取覆盖全局人口的总和
        :param demands_np:
        :return:
        """
        mask=solution_join>self.SMALL_INF
        vj = demands_provider_np[mask, 0, 5:]
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
        providers_np = np.full((demands_provider_np.shape[0], demands_provider_np.shape[1], 2*4), 0.000000001)
        providers_np[:, :, 0] = np.tile(demands_pdd_np[:, 0], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 1] = np.tile(demands_pdd_np[:, 1], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 2] = np.tile(demands_pdd_np[:, 2], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 3] = np.tile(demands_pdd_np[:, 3], (demands_provider_np.shape[0], 1))  # pdd
        providers_np[:, :, 4] = demands_provider_np[:, :, 1 ]  # D_T
        providers_np[:, :, 5] = demands_provider_np[:, :, 2 ]  # D_T
        providers_np[:, :, 6] = demands_provider_np[:, :, 3 ]  # D_T
        providers_np[:, :, 7] = demands_provider_np[:, :, 4]  # D_T
        # 将solution部分增加进去
        mask = solution_join == 0
        mask = mask.reshape(mask.shape[0], 1)
        mask2 = np.tile(mask, (1, demands_provider_np.shape[1]))
        providers_np[:, :, 0][mask2] = 0
        providers_np[:, :, 1][mask2] = 0
        providers_np[:, :, 2][mask2] = 0
        providers_np[:, :, 3][mask2] = 0
        return providers_np

    def calculate_single_provider_gravity_value_np(self, demands_provider_np, demands_pdd_np, solution_join):
        """
        函数：计算重力值
        最费时的运算步骤，预计每一个solution，需要耗时0.5S
        :param demands_np:
        :param DEMANDS_COUNT:
        :return:
        """
        DEMANDS_COUNT = demands_provider_np.shape[1]
        for k in range(0, 4):
            # 执行求取每个需求点的重力值
            for i in range(DEMANDS_COUNT):
                mask = demands_provider_np[:, i, 1+k] <= self.THROD  # D_T在一定有效范围内的
                gravity_value = np.nansum(
                    np.divide(
                        np.add(demands_provider_np[:, i, 0][mask], solution_join[mask]),
                        np.multiply(
                            np.power(demands_provider_np[:, i, 1+k][mask], self.BEITA),  # D_T
                            demands_provider_np[:, i, 5+k][mask]  # vj
                        )
                    )
                )
                demands_pdd_np[i, 4+k] = gravity_value

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
        effective_providers_np = self.create_providers_np(demands_provider_np, demands_pdd_np,solution_join)
        # 因为有四个时刻，所以需要做四次循环
        for k in range(0, 4):
            vj_list = []
            for j in range(effective_providers_np.shape[0]):
                mask = effective_providers_np[j, :, 4+k] <= self.THROD  # 对交通时间做对比
                vj_list.append(
                    np.sum(
                        np.divide(
                            effective_providers_np[j, :, 0+k][mask]   #人口
                            , np.power(effective_providers_np[j, :, 4+k][mask], self.BEITA)   #时间
                        )
                    )
                )
            # 更新vj列
            vj_np = np.array(vj_list).reshape(len(vj_list), 1)
            demands_provider_np[:, :, 5+k] = np.tile(vj_np, (1, demands_provider_np.shape[1]))

    def update_accessibility_to_shp_table(self,demand_id_list, demands_pdd_np_result,shp_table_name,conn,cursor):
        Besty =demands_pdd_np_result[:, 4:8]
        for row in range(Besty.shape[0]):
            update_sql = "update {5} set accessibility08={0},accessibility13={1},accessibility18={2},accessibility22={3} where KEYID={4}".format(
                Besty[row,0], Besty[row,1], Besty[row,2], Besty[row,3], demand_id_list[row],shp_table_name)
            cursor.execute(update_sql)
            conn.commit()
        print("字段更新成功")

    def update_inequlity_to_shp_table(self,demand_id_list, inequlity,shp_table_name,conn,cursor):
        for row in range(inequlity.shape[0]):
            update_sql = "update {5} set inequlity08={0},inequlity13={1},inequlity18={2},inequlity22={3} where KEYID={4}".format(
                inequlity[row,0], inequlity[row,1], inequlity[row,2], inequlity[row,3], demand_id_list[row],shp_table_name)
            cursor.execute(update_sql)
            conn.commit()
        print("字段更新成功")

    def __init__(self, THROD, BEITA,PENTENTIAL_NP_C,OLD_PROVIDERS_COUNT,SMALL_INF):
        self.THROD = THROD
        self.BEITA = BEITA
        self.PENTENTIAL_NP_C=PENTENTIAL_NP_C
        self.OLD_PROVIDERS_COUNT=OLD_PROVIDERS_COUNT
        self.SMALL_INF=SMALL_INF

    def excute_PS_STA(self, demands_provider_np, demands_pdd_np):
        """
        # 主程序
        :param demands_provider_np:
        :param demands_pdd_np:
        :return:
        """
        self.E_OLD_SUM=np.sum(demands_provider_np[:, 0, 0])
        solution = np.full(shape=(self.PENTENTIAL_NP_C,), fill_value=0.0, dtype=np.float32)
        solution_join = np.hstack(
            (np.full(shape=self.OLD_PROVIDERS_COUNT, fill_value=self.SMALL_INF), solution))  # 把前面的已建部分的0.01补齐；

        # demands_provider_np 计算vj
        self.update_provider_vj(demands_provider_np, demands_pdd_np,solution_join)
        self.calculate_single_provider_gravity_value_np(demands_provider_np, demands_pdd_np,solution_join)

        # 居民区 可达性
        # TESTCODE 测试代码 测试可达性计算是不是对的   Test value 值应该和provider的总和一致，如4*10
        self.test_global_accessibility_numpy(demands_pdd_np)

        # # 居民区  计算不公平性数值，该值越小，差异性越小，表示越公平
        inequlity = self.calculate_inequality_np(demands_pdd_np)
        return demands_pdd_np,inequlity

if __name__ == '__main__':
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_jy_time"  # COLLECTION 名称
    BEITA = 1
    # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 0.3  # 有效距离或者时间的阈值
    #### 测试建邺快速充电桩的数据
    DEMANDS_COUNT = 565  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 146
    OLD_PROVIDERS_COUNT = 40
    SMALL_INF=1e-6
    SHP_TABLE_NAME = "pop_grid250_20190412"
    conn = cx_Oracle.connect('powerstation/powerstation@localhost/ORCL')
    cursor = conn.cursor()

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_JY_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list,demand_id_list = mongo_operater_obj.find_records_access_PS_JY_TIME(0, DEMANDS_COUNT,OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C,
                                                                                                       SMALL_INF)  # 必须要先创建索引，才可以执行
    print(demand_id_list)
    access = PS_STA_dynamic(THROD, BEITA,PENTENTIAL_NP_C,OLD_PROVIDERS_COUNT,SMALL_INF)
    demands_pdd_np_result,inequlity=access.excute_PS_STA(demands_provider_np, demands_pdd_np)
    # 将gravity的结果存入到图层上
    access.update_accessibility_to_shp_table(demand_id_list, demands_pdd_np_result, SHP_TABLE_NAME, conn, cursor)
    # 将不公平性的结果存入到图层中
    access.update_inequlity_to_shp_table(demand_id_list, inequlity, SHP_TABLE_NAME, conn, cursor)
    conn.close()



