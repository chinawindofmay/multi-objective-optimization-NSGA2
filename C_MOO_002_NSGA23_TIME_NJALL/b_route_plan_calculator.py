# -*- coding:utf-8 -*-
# 基于高德路径规划API开发服务半径计算方法，计算所有的医院的导航时间和距离，并存储回providers中
import  json,  sys, time, traceback
from b_route_planner import *
from b_rp_queue import RPQueue
from b_save_data_for_rpq import SaverDataOfRPQ
import a_mongo_operater_PS_NJALL_TIME
import math
"""
使用方式，见 CLI 帮助：
python $this_file.py --help
"""

"""
功能配置
"""


DB_NAME="admin"
# COLLECTION_NAME="moo_ps_jy_time"
COLLECTION_NAME="moo_ps_njall_time"

"""
内置函数：程序入口
"""
def do_it(from_row,n_rows,time_slot):
    mongo_operater_obj = a_mongo_operater_PS_NJALL_TIME.MongoOperater(DB_NAME, COLLECTION_NAME)
    n_fetched = 0
    #构建每条记录的请求体
    with SaverDataOfRPQ(mongo_operater_obj) as save_data_of_rpq, RPQueue(time_slot) as route_planner_queue:
        route_planner_queue.bound_save_data_of_rpq(save_data_of_rpq)
        #获取到所有的记录
        rows_json_list = mongo_operater_obj.find_records(from_row,n_rows)
        n_fetched = len(rows_json_list)
        print('{}\tfetched {}'.format(time.asctime(time.localtime(time.time())), n_fetched))
        out_counter=0
        for row_json in rows_json_list:
            providers_json_list=row_json["provider"]
            for item_provider in providers_json_list:
                coordinates = [row_json["x"],row_json["y"],item_provider["x"],item_provider["y"]]
                demand_id = row_json["demand_id"]
                if math.sqrt(math.pow(row_json["x"]-item_provider["x"],2)+math.pow(row_json["y"]-item_provider["y"],2))*111.3>40:
                    out_counter+=1
                    _update_query_one_provider_per_time(mongo_operater_obj,demand_id,item_provider,time_slot)
                    continue
                route_planner_queue += Drivinger(demand_id,item_provider, *coordinates)                   #将每条记录请求体放入请求队列对象中。对于+=号操作，可变对象调用__add__，不可变对象调用的是__iadd__
    print(out_counter ) #390651 873*849-390651，显然没有超,30万范围
    return n_fetched

def _update_query_one_provider_per_time(mongo_operater_obj, demand_id,provider, time_slot):
    #获得预先已经存在provider及其travel
    travel=provider["travel"]
    travel["D_T"+time_slot]=100
    #更新一条记录
    mongo_operater_obj.update_provider_travel_record(travel, demand_id, provider["provider_id"])
    print('{}\t out---demand point is {}, provider point is {}'.format(time.asctime(time.localtime(time.time())),demand_id, provider["provider_id"]))

# TIME_LIST = ["21"]
TIME_LIST = ["08","13", "18", "22"]

# , "13", "18", "22"

"""
内置函数：时间控制
"""
def time_controller():
    current_time = time.strftime("%H", time.localtime())
    print(current_time)
    if current_time in TIME_LIST:
        TIME_LIST.remove(current_time)
        return current_time
    else:
        return None

###整个py文件的主入口
if __name__ == "__main__":
    """
    入口的运行逻辑：
    首先，定义了1时间点及对应表的dictionary
    其次，开始执行time循环，当达到时间点时，POP出一个
    然后，开始遍历其中的两张表，
    第四，表名会传入到do_it中，便于Select和update，
    第五，执行请求过程中会有批量请求和非批量请求，决定是否批量的缘故在于高德接口是否支持。
    第六，完成每张表的请求后，还会调用STD计算，处理路径规划表，找出最快的时间，最近的交通距离，最小的STD，及对应的交通方式
    """
    while len(TIME_LIST) > 0:
        time_slot=time_controller()
        if time_slot != None:
            # 进入处理逻辑
            fromrow = 0
            n_rows = 873
            #执行批量请求
            n_rows_fatch = do_it(fromrow, n_rows,time_slot)
            print('have affected {0} at {1}'.format(n_rows_fatch, time.asctime(time.localtime(time.time()))))
        time.sleep(60)