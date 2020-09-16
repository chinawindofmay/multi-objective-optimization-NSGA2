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

        # 遇到如下异常：
        # TypeError:
        # if no direction is specified, key_or_list must be an instance of list
        # 解决方法：
        # db.test.find().sort([("name", 1), ("age", 1)])
        # 原因：在python中只能使用列表进行排序，不能使用字典
        # db.c1.find().sort({"demand_id":1}).skip(0).limit(1)
        records=self.collection.find().sort([("demand_id",1)])
        # records=self.collection.find({},{"_id":1,"demand_id":1}).sort([("demand_id",1)])
        #这里是导致耗时最大的地方，需要采用pandas来readjson
        records_list = list(records)[begin_counter:end_counter]

        # records_list_pd = pd.DataFrame(records_list)    ####这里也存在多级，暂时没有转化
        # #records_length = self.collection.find().sort([("demand_id", 1)]).limit(250).skip(250).count(True)   要有true，不然返回的是整体
        return records_list


    """
    函数：基于Numpy存储，查找特定位置段的对象集合，跳过skip_count个，找出splic_count个
    """
    def find_records_format_in_numpy_gravity(self, begin_counter, DEMANDS_COUNT, PROVIDERS_COUNT):
        records_list = list(self.collection.find().sort([("demand_id", 1)]))[begin_counter:DEMANDS_COUNT]
        demands_np=np.full((DEMANDS_COUNT,PROVIDERS_COUNT,4,3),0.000000001)
        # demands_np=np.full((2265,26,4,3),0.000000001)
        for i in range(DEMANDS_COUNT):
            demands_np[i,:,:,0]=records_list[i]["population"]
            demands_np[i, :, :, 1]=0.0001
            for j in range(PROVIDERS_COUNT):
                demands_np[i, j, 0, 2] = records_list[i]["provider"][j]["quickcharge"]
                demands_np[i, j, 1, 2]= records_list[i]["provider"][j]['travel']["D_T"]
        return demands_np


    """
    函数：更新特定demand_id值的demand对象的特定的provider_id_value值的provider对象的Travel值
    """
    def update_provider_travel_record(self, travel, demand_id, provider_id_value):
        self.collection.update({"demand_id":demand_id,'provider.provider_id': provider_id_value}, {"$set": {'provider.$.travel': travel}})

