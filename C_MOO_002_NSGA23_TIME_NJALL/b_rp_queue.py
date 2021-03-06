# -*- coding:utf-8 -*-
import requests,json,traceback,time
from b_route_planner import *
# API_KEY = 'ccf0b26003c9f5b55ce2c47f1ac67bdb'
API_KEY = '87874adcc8ba2083e2497d17e937eaee'
# API_KEY = '234e96d7ab5d31365ddd32c213cffb7b'


"""
类：请求对象的ADD队列，ADD进来之后根据是否是BATCH来触发：批量请求、非批量请求封装和吞吐量控制功能
"""
class RPQueue(object):
    BATCH_SIZE = 20
    API_SITE = 'http://restapi.amap.com'
    BATCH_URL = '/v3/batch'
    BATCH_URI = '{}{}?key={}'.format(API_SITE, BATCH_URL, API_KEY)
    BATCH_HDR = {'Content-Type': 'application/json;charset=utf-8'}
    REQ_INTERVAL = 0.01  # throughput < 200Hz

    def __init__(self,time_slot):
        self.request_cache = []
        self.on_each_ok_callback_save_data = None
        self.last_req_time = 0.0
        self.time_slot=time_slot

    """
    建议用在 with 语句里使用本对象
    """
    def __enter__(self):
        return self

    """
    结束前清空队列缓存
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._batch_request_of_rpqueue(self.request_cache,time_slot=self.time_slot)

    """
    加入队列
    """

    def __iadd__(self, route_planner_object):
        if not isinstance(route_planner_object, BaseRoutePlanner.CONCRETE_TYPES):
            return self
        #判别是否是批量请求对象
        if route_planner_object.batchable:
            self.request_cache.append(route_planner_object)
            #判别是否达到BATCH_SIZE个数，如果达到，则只需_batch_request，完成执行后清空，request_cache.clear()
            if len(self.request_cache) == type(self).BATCH_SIZE:
                self._batch_request_of_rpqueue([r for r in self.request_cache],time_slot=self.time_slot)
                self.request_cache.clear()
        else:
            self._standalone_request_of_rpqueue(route_planner_object,time_slot=self.time_slot)
        return self

    """
    绑定请求成功后的回调函数，用来处理存储数据的逻辑
    """
    def bound_save_data_of_rpq(self, callback):
        self.on_each_ok_callback_save_data = callback

    """
    函数：发送非批量请求
    """
    def _standalone_request_of_rpqueue(self, route_planner_object,time_slot):
        cls = type(self)
        self._throughput_control_of_rpqueue(cls.REQ_INTERVAL)
        try:
            #发生请求的触发代码
            reply = requests.get(cls.API_SITE + route_planner_object.uri, timeout=1)
            if 200 <= reply.status_code < 300:
                #此处直接调用了类的__call__方法
                self.on_each_ok_callback_save_data(reply.json(), route_planner_object,time_slot)
        except:
            # traceback.print_exc()
            print(cls.API_SITE + route_planner_object.uri, "   time out")


    """
    函数：发送批量请求
    """
    def _batch_request_of_rpqueue(self, route_planner_objects,time_slot):
        ops = [dict(url=r.uri) for r in route_planner_objects]
        cls = type(self)

        #吞吐量控制，睡一会
        self._throughput_control_of_rpqueue(cls.REQ_INTERVAL * cls.BATCH_SIZE)
        req=""
        res=""
        try:
            reply = requests.post(cls.BATCH_URI, data=json.dumps(dict(ops=ops)),headers=cls.BATCH_HDR,timeout=3)
            if 200 <= reply.status_code < 300:
                for route_planner_object, route_planner_response in zip(route_planner_objects, reply.json()):
                    if 'status' in route_planner_response:
                        if 200 <= route_planner_response['status'] < 300 and self.on_each_ok_callback_save_data is not None:
                            try:
                                # 此处直接调用了类的__call__方法，请求成功后的被调用，用来数据入库，先填充POI_Attr，填充到一定数量之后，就存数据库。
                                self.on_each_ok_callback_save_data(route_planner_response['body'], route_planner_object,time_slot)
                            except:
                                traceback.print_exc()
        except:
            # traceback.print_exc()
            print("batch  requests.post fail")
            for route_planner_object in route_planner_objects:
                self.on_each_ok_callback_save_data(route_planner_response_body=None,route_planner_object=route_planner_object,batch_fail=True,time_slot=time_slot)
    """
    函数：吞吐量控制
    """
    def _throughput_control_of_rpqueue(self, min_interval):
        now_secs = float(time.time())
        remaining_req_interval = self.last_req_time + min_interval - now_secs
        if remaining_req_interval > 0:
            time.sleep(remaining_req_interval)
            self.last_req_time = float(time.time())
        else:
            self.last_req_time = now_secs