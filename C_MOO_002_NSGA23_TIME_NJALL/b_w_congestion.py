#coding:utf-8
import sys
import math
import numpy as np
import cx_Oracle
import json
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks",context="notebook")
import a_mongo_operater_PS_NJALL_TIME

def calculate_change_value(array_mtm):
    if np.all(array_mtm < 100):
        sum_value=0
        count=0
        for item in array_mtm[0:-1]:
            if item >array_mtm[-1]:
                sum_value+=(item-array_mtm[-1])/array_mtm[-1]
                count+=1
        return 0 if count==0 else sum_value/count
    else:
        return 0

def show_congestion_line_chart(demands_provider_np):
    congistion_D_list=[]
    for i in range(demands_provider_np.shape[0]):
        for j in range(demands_provider_np.shape[1]):
            pt_array_D = np.array([demands_provider_np[i,j,1],
                                    demands_provider_np[i,j,2],
                                    demands_provider_np[i,j,3],
                                    demands_provider_np[i,j,4]])
            congistion_D = calculate_change_value(pt_array_D)
            congistion_D_list.append(congistion_D)

    # x_data = [i for i in range(len(congistion_PT_list))]
    # plt.plot(x_data, congistion_D_list, color='red', linewidth=3.0, linestyle='-.')
    # plt.show()

    congistion_d_np=np.array(congistion_D_list)
    result_np_array=congistion_d_np[congistion_d_np[:]>0]
    # # st_rt_np_ay=np.sort(a=result_np_array,axis=0,order=0)   所有列一起排序
    # result_np_array = result_np_array[np.argsort(result_np_array[:, 0])]   #按某一列排序

    # breaks_D = jenkspy.jenks_breaks(congistion_D_list, nb_class=4)
    # print(breaks_D)
    x_data = [i for i in range(result_np_array.shape[0])]
    plt.plot(x_data, result_np_array[:], color='grey', linewidth=1.0, alpha=0.9, linestyle='-.',label="Driving")
    plt.tick_params(labelsize=15)  # 刻度字体大小13
    plt.ylabel("Traffic congestion coefficient", fontdict={'family': 'Times New Roman', 'size': 16,'fontweight':'bold'})
    plt.xlabel("OD flows series", fontdict={'family': 'Times New Roman', 'size': 16,'fontweight':'bold'})
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    # plt.savefig('./congestion.jpg', dpi=200)
    plt.show()

if __name__=="__main__":
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_njall_time"  # COLLECTION 名称
    BEITA = 1
    # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 0.5  # 有效距离或者时间的阈值
    #### 测试建邺快速充电桩的数据
    DEMANDS_COUNT = 873  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 570  # 候选点570
    OLD_PROVIDERS_COUNT = 279  # 已有快速充电站279个
    SMALL_INF = 1e-6
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_NJALL_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_provider_np, demands_pdd_np, provider_id_list, demand_id_list = mongo_operater_obj.find_records_access_PS_NJALL_TIME(0, DEMANDS_COUNT, OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C,SMALL_INF)  # 必须要先创建索引，才可以执行
    show_congestion_line_chart(demands_provider_np)