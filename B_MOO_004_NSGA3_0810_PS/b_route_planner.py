# -*- coding:utf-8 -*-
"""
基类：路径规划基础对象
作用：存储URL格式、控制是否是批量请求的方式、解析返回结果的解析逻辑
"""

class BaseRoutePlanner(object):
    URL = ''
    def __init__(self, demand_id,provider, *args):
        self.API_KEY = "234e96d7ab5d31365ddd32c213cffb7b"
        # self.API_KEY = "ccf0b26003c9f5b55ce2c47f1ac67bdb"
        # self.API_KEY = "87874adcc8ba2083e2497d17e937eaee"
        #type用来取实例的类。访问静态变量用的。
        cls = type(self)
        self.demand_id = demand_id
        self.provider=provider
        self.batchable = True
        self.uri = '{}?key={}&origin={},{}&destination={},{}'.format(cls.URL, self.API_KEY, *args)
    """
    函数：解析返回结果的解析逻辑
    通用的会有：步行和开车；
    Warlkinger方式，会出现OVER_DIRECTION_RANGE，表示距离太远了，所以规划路径失败
    """
    def parse_response_body_get_time(self, response_body):
        # 避免Warlkinger方式，会出现OVER_DIRECTION_RANGE，表示距离太远了，所以规划路径失败
        if 'route' in response_body.keys():
            paths = response_body['route']['paths']
            if paths is None or len(paths) == 0:
                return dict(time=100)
            else:
                path_0 = paths[0]
                return dict(time=round(float(path_0['duration']) / 3600, 2))
        else:
            return dict(time=100)


"""
标准类：驾车路径规划对象
作用：存储驾车的URL格式、是批量请求的方式、解析返回结果逻辑与基类一致
"""
class Drivinger(BaseRoutePlanner):
    URL = '/v3/direction/driving'

#增加类属性
BaseRoutePlanner.CONCRETE_TYPES = (Drivinger)

