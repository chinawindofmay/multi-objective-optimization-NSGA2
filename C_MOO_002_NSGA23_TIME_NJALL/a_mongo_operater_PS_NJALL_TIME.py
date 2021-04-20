#coding:utf-8
from pymongo import MongoClient
import json
import numpy as np
import pandas as pd
import math

class MongoOperater:
    def __init__(self,db_name,collection_name):
        self.client = MongoClient(host='localhost', port=27017)
        self.db = self.client[db_name]
        self. collection = self.db[collection_name]
    """
    函数：关闭
    """
    def close(self):
        self.client.close()

    """
        函数：插入一个对象到数据库中
    """
    def insert_record(self,json_object):
        result = self.collection.insert(json_object)
        print("OK")

    """
        函数：查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records(self,begin_counter,end_counter):
        #1是升序
        #skip()与limit()的前后顺序没有要求，不管怎么放置他们执行的顺序都是先sort()后skip()最后limit()
        # 需要创建索引：Sort operation used more than the maximum 33554432 bytes of RAM. Add an index, or specify a smaller
        # db.yourCollection.createIndex({ < field >: < 1 or -1 >})
        # db.yourCollection.getIndexes() // 查看当前collection的索引
        self.collection.create_index([("demand_id",1)])
        # 遇到如下异常：
        # TypeError:
        # if no direction is specified, key_or_list must be an instance of list
        # 解决方法：
        # db.test.find().sort([("name", 1), ("age", 1)])
        # 原因：在python中只能使用列表进行排序，不能使用字典
        # db.c1.find().sort({"demand_id":1}).skip(0).limit(1)
        # records=self.collection.find({},{"_id":1,"demand_id":1}).sort([("demand_id",1)])
        #这里是导致耗时最大的地方，需要采用pandas来readjson
        records_list = list(self.collection.find().sort([("demand_id",1)]))[begin_counter:end_counter]

        # records_list_pd = pd.DataFrame(records_list)    ####这里也存在多级，暂时没有转化
        # #records_length = self.collection.find().sort([("demand_id", 1)]).limit(250).skip(250).count(True)   要有true，不然返回的是整体
        return records_list


    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records_format_numpy_1(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT):
        self.collection.create_index([("demand_id",1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        demands_np=np.full((DEMANDS_COUNT,PROVIDERS_COUNT,4,3),0.000000001)
        # demands_np=np.full((2265,26,4,3),0.000000001)
        for i in range(DEMANDS_COUNT):
            demands_np[i, :, :, 0]=records_list[i]["population"]
            demands_np[i, :, :, 1]=0.0001
            for j in range(PROVIDERS_COUNT):
                demands_np[i, j, 0, 2] = records_list[i]["provider"][j]["quickcharge"]
                demands_np[i, j, 1, 2]= records_list[i]["provider"][j]['travel']["D_T"]
        return demands_np

    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records_format_np_theory(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 3), 0.000000001)
        demands_pdd_np = np.full(( DEMANDS_COUNT, 2), 0.000000001)
        # demands_provider_np=np.full((89,184,3),0.000000001)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] = records_list[i]["population"]
            demands_provider_np[:,i, 2] = 0.0001    #vj
            for j in range(PROVIDERS_COUNT):
                demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["ps"]
                demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T"]
        #用于后面将结果做展示
        provider_id_list=[]
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        return demands_provider_np,demands_pdd_np,provider_id_list

    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records_format_np_jianye(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 3), 0.000000001)
        demands_pdd_np = np.full((DEMANDS_COUNT, 2), 0.000000001)
        # demands_provider_np=np.full((89,184,3),0.000000001)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] = records_list[i]["population"]
            demands_provider_np[:, i, 2] = 0.0001  # vj
            for j in range(PROVIDERS_COUNT):
                demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["quickcharge"]
                demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T"]
        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        return demands_provider_np, demands_pdd_np,provider_id_list

    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records_PS_NJALL_TIME(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT, SMALL_INF):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        # 片，行，列 这个是考虑了未建和已建的
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 9), SMALL_INF)
        # 行，列
        demands_pdd_np = np.full((DEMANDS_COUNT, 8), SMALL_INF)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] =  records_list[i]["populist"][7] / 10000*118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 1] = records_list[i]["populist"][13] / 10000*118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 2] = records_list[i]["populist"][17] / 10000*118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 3] =  records_list[i]["populist"][21] / 10000*118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            for j in range(PROVIDERS_COUNT):
                if records_list[i]["provider"][j]["pszcount"]==0 and records_list[i]["provider"][j]["type"]=="gjd":
                    demands_provider_np[j, i, 0] =2 #对于已建但是规模为0的供给点，做一个赋值，附上一个最小值，因为如果最小值变为0了，会导致排队时间无限大；
                else:
                    demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["pszcount"]

                #获取交通阻尼时间
                if records_list[i]["provider"][j]['travel']["D_T08"] < 0.001:
                    demands_provider_np[j, i, 1] = 0.1
                else:
                    demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T08"]
                if records_list[i]["provider"][j]['travel']["D_T13"] < 0.001:
                    demands_provider_np[j, i, 2] = 0.1
                else:
                    demands_provider_np[j, i, 2] = records_list[i]["provider"][j]['travel']["D_T13"]
                if records_list[i]["provider"][j]['travel']["D_T18"] < 0.001:
                    demands_provider_np[j, i, 3] = 0.1
                else:
                    demands_provider_np[j, i, 3] = records_list[i]["provider"][j]['travel']["D_T18"]
                if records_list[i]["provider"][j]['travel']["D_T21"] < 0.001:
                    demands_provider_np[j, i, 4] = 0.1
                else:
                    demands_provider_np[j, i, 4] = records_list[i]["provider"][j]['travel']["D_T21"]

                # if records_list[i]["provider"][j]['travel']["D_T21"] < 0.001:
                #     demands_provider_np[j, i, 1] = 0.1
                #     demands_provider_np[j, i, 2] = 0.1
                #     demands_provider_np[j, i, 3] = 0.1
                #     demands_provider_np[j, i, 4] = 0.1
                # else:
                #     demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T21"]
                #     demands_provider_np[j, i, 2] = records_list[i]["provider"][j]['travel']["D_T21"]
                #     demands_provider_np[j, i, 3] = records_list[i]["provider"][j]['travel']["D_T21"]
                #     demands_provider_np[j, i, 4] = records_list[i]["provider"][j]['travel']["D_T21"]

        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        # 测试代码
        # # 判断是否D_T存在有零的情况
        # print("check data time is valid",np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        # mask =demands_provider_np[:, :, 1:5] < 0.001
        # np.where(mask,demands_provider_np[:, :, 1:5],0.2111)
        # print("check data time is valid", np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        return demands_provider_np, demands_pdd_np, provider_id_list

    """
        函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
        """

    def find_records_PS_NJALL_TIME_poplimit_dynamic(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT, SMALL_INF):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        # 片，行，列 这个是考虑了未建和已建的
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 9), SMALL_INF)
        # 行，列
        demands_pdd_np = np.full((DEMANDS_COUNT, 8), SMALL_INF)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] = records_list[i]["populist"][7] / 10000 * (850/1294) * 118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 1] = records_list[i]["populist"][12] / 10000 * (850/1088)* 118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 2] = records_list[i]["populist"][17] / 10000 * (850/1320)* 118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 3] = records_list[i]["populist"][21] / 10000 * (850/877)* 118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            for j in range(PROVIDERS_COUNT):
                if records_list[i]["provider"][j]["pszcount"] == 0 and records_list[i]["provider"][j]["type"] == "gjd":
                    demands_provider_np[j, i, 0] = 2  # 对于已建但是规模为0的供给点，做一个赋值，附上一个最小值，因为如果最小值变为0了，会导致排队时间无限大；
                else:
                    demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["pszcount"]
                # 获取交通阻尼时间
                if records_list[i]["provider"][j]['travel']["D_T08"] < 0.001:
                    demands_provider_np[j, i, 1] = 0.1
                else:
                    demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T08"]
                if records_list[i]["provider"][j]['travel']["D_T13"] < 0.001:
                    demands_provider_np[j, i, 2] = 0.1
                else:
                    demands_provider_np[j, i, 2] = records_list[i]["provider"][j]['travel']["D_T13"]
                if records_list[i]["provider"][j]['travel']["D_T18"] < 0.001:
                    demands_provider_np[j, i, 3] = 0.1
                else:
                    demands_provider_np[j, i, 3] = records_list[i]["provider"][j]['travel']["D_T18"]
                if records_list[i]["provider"][j]['travel']["D_T21"] < 0.001:
                    demands_provider_np[j, i, 4] = 0.1
                else:
                    demands_provider_np[j, i, 4] = records_list[i]["provider"][j]['travel']["D_T21"]

        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        # 测试代码
        # # 判断是否D_T存在有零的情况
        # print("check data time is valid",np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        # mask =demands_provider_np[:, :, 1:5] < 0.001
        # np.where(mask,demands_provider_np[:, :, 1:5],0.2111)
        # print("check data time is valid", np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        return demands_provider_np, demands_pdd_np, provider_id_list


    def find_records_PS_NJALL_TIME_poplimit_static(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT, SMALL_INF):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        # 片，行，列 这个是考虑了未建和已建的
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 3), SMALL_INF)
        # 行，列
        demands_pdd_np = np.full((DEMANDS_COUNT, 2), SMALL_INF)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] = (records_list[i]["populist"][7] / 10000 * (850/1294.2254) * 118 + \
                                   records_list[i]["populist"][12] / 10000 * (850/1088.7428)* 118 + \
                                   records_list[i]["populist"][17] / 10000 * (850 / 1320.9574) * 118 + \
                                   records_list[i]["populist"][21] / 10000 * (850 / 8777.072) * 118)/4
                                    # 单位换成万人，万人拥有的新能源汽车量是118辆
            for j in range(PROVIDERS_COUNT):
                if records_list[i]["provider"][j]["pszcount"] == 0 and records_list[i]["provider"][j]["type"] == "gjd":
                    demands_provider_np[j, i, 0] = 2  # 对于已建但是规模为0的供给点，做一个赋值，附上一个最小值，因为如果最小值变为0了，会导致排队时间无限大；
                else:
                    demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["pszcount"]

                # 获取交通阻尼时间
                if records_list[i]["provider"][j]['travel']["D_T08"] < 0.001:
                    t1= 0.1
                else:
                    t1 = records_list[i]["provider"][j]['travel']["D_T08"]
                if records_list[i]["provider"][j]['travel']["D_T13"] < 0.001:
                    t2 = 0.1
                else:
                    t2 = records_list[i]["provider"][j]['travel']["D_T13"]
                if records_list[i]["provider"][j]['travel']["D_T18"] < 0.001:
                    t3 = 0.1
                else:
                    t3 = records_list[i]["provider"][j]['travel']["D_T18"]
                if records_list[i]["provider"][j]['travel']["D_T21"] < 0.001:
                    t4 = 0.1
                else:
                   t4 = records_list[i]["provider"][j]['travel']["D_T21"]
                demands_provider_np[j, i, 1]=(t1+t2+t3+t4)/4

        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        # 测试代码
        # # 判断是否D_T存在有零的情况
        # print("check data time is valid",np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        # mask =demands_provider_np[:, :, 1:5] < 0.001
        # np.where(mask,demands_provider_np[:, :, 1:5],0.2111)
        # print("check data time is valid", np.count_nonzero(demands_provider_np[:, :, 1:5] < 0.001))
        return demands_provider_np, demands_pdd_np, provider_id_list

    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    为了计算动态可达性和可视化而建立的函数，主要区别于上面的是增加了demand_id_list
    """
    def find_records_access_PS_NJALL_TIME(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT, SMALL_INF):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        # 片，行，列
        demands_provider_np = np.full((PROVIDERS_COUNT, DEMANDS_COUNT, 9), SMALL_INF)
        # 行，列
        demands_pdd_np = np.full((DEMANDS_COUNT, 8), SMALL_INF)
        for i in range(DEMANDS_COUNT):
            demands_pdd_np[i, 0] = records_list[i]["populist"][7] / 10000*118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 1] = records_list[i]["populist"][13] / 10000 *118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 2] = records_list[i]["populist"][17] / 10000 *118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            demands_pdd_np[i, 3] = records_list[i]["populist"][21] / 10000 *118  # 单位换成万人，万人拥有的新能源汽车量是118辆
            for j in range(PROVIDERS_COUNT):
                demands_provider_np[j, i, 0] = records_list[i]["provider"][j]["pszcount"]
                # 昆山特有 重复在一起的点位
                if records_list[i]["provider"][j]['travel']["D_T08"] < 0.001:
                    demands_provider_np[j, i, 1] = 0.1
                else:
                    demands_provider_np[j, i, 1] = records_list[i]["provider"][j]['travel']["D_T08"]

                if records_list[i]["provider"][j]['travel']["D_T13"] < 0.001:
                    demands_provider_np[j, i, 2] = 0.1
                else:
                    demands_provider_np[j, i, 2] = records_list[i]["provider"][j]['travel']["D_T13"]

                if records_list[i]["provider"][j]['travel']["D_T18"] < 0.001:
                    demands_provider_np[j, i, 3] = 0.1
                else:
                    demands_provider_np[j, i, 3] = records_list[i]["provider"][j]['travel']["D_T18"]

                if records_list[i]["provider"][j]['travel']["D_T21"] < 0.001:
                    demands_provider_np[j, i, 4] = 0.1
                else:
                    demands_provider_np[j, i, 4] = records_list[i]["provider"][j]['travel']["D_T21"]
        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        demand_id_list = []
        for i in range(DEMANDS_COUNT):
            demand_id_list.append(records_list[i]["demand_id"])
        #测试代码 判断是否D_T存在有零的情况，为0则表示没有，不为零则表示有问题
        print(np.sum(demands_provider_np[:, :, 1:5]<=0),"-------测试代码 判断是否D_T存在有零的情况，为0则表示没有，不为零则表示有问题")
        return demands_provider_np, demands_pdd_np, provider_id_list,demand_id_list

    def find_records_travel_time_NJALL(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT, SMALL_INF):
        self.collection.create_index([("demand_id", 1)])
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        # 片，行，列
        travel_time = []
        for i in range(DEMANDS_COUNT):
            for j in range(PROVIDERS_COUNT):
                if records_list[i]["provider"][j]['travel']["D_T08"]<0.5:
                    travel_obj=[]
                    travel_obj.append(records_list[i]["x"])
                    travel_obj.append(records_list[i]["y"])
                    travel_obj.append(records_list[i]["provider"][j]['x'])
                    travel_obj.append(records_list[i]["provider"][j]['y'])
                    if records_list[i]["provider"][j]['travel']["D_T08"] < 0.001:
                        travel_obj.append(0.1)
                    else:
                        travel_obj.append(records_list[i]["provider"][j]['travel']["D_T08"])
                    travel_time.append(travel_obj)
        # 用于后面将结果做展示
        provider_id_list = []
        for j in range(PROVIDERS_COUNT):
            provider_id_list.append(records_list[0]["provider"][j]["provider_id"])
        demand_id_list = []
        for i in range(DEMANDS_COUNT):
            demand_id_list.append(records_list[i]["demand_id"])
        return travel_time,  provider_id_list,demand_id_list

    """
    函数：更新特定demand_id值的demand对象的特定的provider_id_value值的provider对象的Travel值
    """
    def update_provider_travel_record(self, travel, demand_id, provider_id_value):
        self.collection.update({"demand_id":demand_id,'provider.provider_id': provider_id_value}, {"$set": {'provider.$.travel': travel}})
