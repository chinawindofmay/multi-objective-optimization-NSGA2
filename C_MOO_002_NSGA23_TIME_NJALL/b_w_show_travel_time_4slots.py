
import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib
import a_mongo_operater_PS_NJALL_TIME
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid', context='notebook')
# sns.set(style="ticks", color_codes=True)
import pandas as pd



if __name__ == '__main__':
    DB_NAME = "admin"  # MONGODB数据库的配置
    COLLECTION_NAME = "moo_ps_njall_time"  # COLLECTION 名称
    BEITA = 1
    # 主要参考A multi-objective optimization approach for health-care facility location-allocation problems in highly developed cities such as Hong Kong
    THROD = 0.5  # 有效距离或者时间的阈值
    #### 测试建邺快速充电桩的数据
    DEMANDS_COUNT = 873  # 需求点，即小区，个数
    # 供给点，即充电桩，的个数，与X_num2保持一致
    PENTENTIAL_NP_C = 570   #候选点570
    OLD_PROVIDERS_COUNT = 279  #已有快速充电站279个
    SMALL_INF=1e-6
    PROVIDERS_COUNT=OLD_PROVIDERS_COUNT + PENTENTIAL_NP_C

    TIME_SLOTS=["D_T08","D_T13","D_T18","D_T22"]

    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater_PS_NJALL_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    for time_slot in TIME_SLOTS:
        print("正在处理{0}，请等待".format(time_slot))
        # 获取到所有的记录
        demands_provider_np, provider_id_list,demand_id_list = mongo_operater_obj.find_records_travel_time_NJALL(0, DEMANDS_COUNT, PROVIDERS_COUNT,SMALL_INF,time_slot)  # 必须要先创建索引，才可以执行

        ODs = pd.DataFrame(data=demands_provider_np,columns=["dx","dy","px","py","t"])

        ODs.to_csv("./log/tzzs_data_{0}.csv".format(time_slot))
        print("处理完毕")


