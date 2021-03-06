# -*- coding:utf-8 -*-

import json,  time, traceback
from b_route_planner import *

"""
类：高德 API 响应结果处理队列，提供数据入库功能
"""
class SaverDataOfRPQ(object):
    BUFFER_HIGH_WATER = 200
    def __init__(self, mongo_operater_obj):
        #存放结果值，每次存入数据库一次，就删掉一个
        self.result_cache_dict = {}
        #当按demand的方式提交的时候，需要使用的IobjectID 的计数器，和self.result_cache_dict配合使用。
        self.result_cache_iobjectid_counter_dict={}

        self.mongo_operater_obj = mongo_operater_obj


        self.POI_ATTR_DICT = {
            (Drivinger.__name__, 'time'): "D_T"
        }

        self.cursor = None
        self.n_accepted = 0
        self.n_skipped = 0


    """
    函数：建议用在 with 语句里使用本对象
    """
    def __enter__(self):
        return self

    """
    函数：结束前清空队列缓存
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mongo_operater_obj.close()


    """
    函数：请求成功后的被调用，用来数据入库
    默认参数，又称为缺省参数，一般放在最后面；
    """
    def __call__(self, route_planner_response_body, route_planner_object,time_slot,batch_fail=False):
        result={}
        #第一步：判断batch是否成功了？
        if batch_fail:
            result={"time":100}
        else:
            # 第一部分：更新result_cache_dict的值
            # result的基本形式是：{"distance":11,"time":1.4}
            result = route_planner_object.parse_response_body_get_time(route_planner_response_body)

        #第二步：获取一些基本的值
        demand_id = route_planner_object.demand_id
        provider = route_planner_object.provider
        provider_id=provider["provider_id"]
        one_intact_rp_id = ("-").join([str(demand_id),str(provider_id)])

        #第三步：如果之前没有提交one_intact_rp_id类型的，则新建一个属性键值对
        if one_intact_rp_id not in self.result_cache_dict:
            self.result_cache_dict[one_intact_rp_id] = {}

        #第四部分：判别是否需要做提交方式
        #逐个供给点进行提交，这种方式在不停的读写数据库，更新provider_08等字段
        # 给POI_attr_dict添加4种交通方式的属性值
        self.result_cache_dict[one_intact_rp_id][type(route_planner_object).__name__] = result
        poi_attr_dict = self.result_cache_dict[one_intact_rp_id]
        # print("this is a test. The objectid is {0}, the provider_id is {1}, the travel mode is {2}".format(demand_id,provider["provider_id"],type(route_planner_object).__name__))
        if len(poi_attr_dict) == 1:###一种Drivinger类型
            #更新到数据库
            self._update_query_one_provider_per_time(demand_id, provider, poi_attr_dict,time_slot)
            # 删除掉已经全部入库的记录了
            del self.result_cache_dict[one_intact_rp_id]


    """
    服务__call__的内置函数
    获取符合travel_model_count*provider_count数量要求的ObjectID对应的记录
    """
    def get_demand_records(self,result_cache,result_cache_iobjectid_counter_dict,travel_model_count,provider_count):
        poi_attr_dict_dict={}
        demand_id_full=None
        for key_demand_id,value_demand_id in result_cache_iobjectid_counter_dict.items():
            if value_demand_id==travel_model_count*provider_count:
                demand_id_full=key_demand_id
                for key in list(result_cache.keys()):
                    if (str(key_demand_id) in key):
                        if len(result_cache[key])==len(BaseRoutePlanner.CONCRETE_TYPES):
                            poi_attr_dict_dict[key]=result_cache[key]
                        else:
                            continue
                    else:
                        continue
            else:
                continue
        return (demand_id_full,poi_attr_dict_dict)

    """
    服务__call__的内置函数
    删除已经存储到数据库中result_cache和result_cache_iobjectid_counter_dict对象
    """
    def delete_demand_records_and_idemand_id_full(self,result_cache,result_cache_iobjectid_counter_dict,demand_id_full):
        #del的时候不能用 for key,values in result_cache.items(): 因为这个是基于Iteration的，会报错RuntimeError: dictionary changed size during iteration
        #大概是说字典在遍历时不能进行修改，建议转成列表或集合处理。
        #需要使用：
        for key in list(result_cache.keys()):
            if str(demand_id_full) in key :
                if len(result_cache[key])==len(BaseRoutePlanner.CONCRETE_TYPES):
                    del result_cache[key]
        del result_cache_iobjectid_counter_dict[demand_id_full]

    """
    函数：每一个provider，提交一次；
    """
    def _update_query_one_provider_per_time(self, demand_id,provider, poi_attr,time_slot):
        #获得预先已经存在provider及其travel
        travel=provider["travel"]
        travel["D_T"+time_slot]=poi_attr["Drivinger"]["time"]
        #更新一条记录
        self.mongo_operater_obj.update_provider_travel_record(travel, demand_id, provider["provider_id"])
        print('{}\tcommitted---demand point is {}, provider point is {}'.format(time.asctime(time.localtime(time.time())),demand_id, provider["provider_id"]))



