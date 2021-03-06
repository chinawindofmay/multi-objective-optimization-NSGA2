# -*- coding:utf-8 -*-

"""
用于初始化provider对象
"""

import math
import json
#ORACLE库
import cx_Oracle
from pymongo import MongoClient
import a_mongo_operater_PS_NJALL_TIME
import copy

import zlib

conn_string='powerstation/powerstation@192.168.90.2/ORCL'

def initial_providers(pianyi_x_ave,pianyi_y_ave):
    provider_list=[]
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    # 第一步:已经建成的，不做调整
    select_providers_sql = '''select keyid_ps, x , y,pscount,t5.sum_快充+t5.sum_慢充 as pszcount , xzqmc as address from POPU0412_JOIN_PARKS_JOIN_PS_PO t5 where t5.pscount>0  order by  id1 asc '''
    # 执行
    cursor.execute(select_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]+pianyi_x_ave
            y = row[2]+pianyi_y_ave
            pscount = row[3]
            pszcount = row[4]
            address = row[5]

            provider={}
            provider["provider_id"]=key_id
            provider["x"]=x
            provider["y"]=y
            provider["pscount"]=pscount
            provider["pszcount"]=pszcount
            provider["address"]=address
            provider["travel"]={"D_T":999}
            provider["type"] = "gjd"
            provider_list.append(provider)
        except:
            continue

    # 第2步:未建的，要做增加
    select_petential_providers_sql = '''
select keyid_parks, x , y,0 as pscount,0 as pszcount , t5.parkscount||xzqmc as address from POPU0412_JOIN_PARKS_JOIN_PS_PO t5 where t5.parkscount>0  order by  id1 asc 
 '''
    # 执行
    cursor.execute(select_petential_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]+pianyi_x_ave
            y = row[2]+pianyi_y_ave
            pscount = row[3]
            pszcount = row[4]
            address = row[5]

            provider = {}
            provider["provider_id"] = key_id
            provider["x"] = x
            provider["y"] = y
            provider["pscount"] = pscount
            provider["pszcount"] = pszcount
            provider["address"] = address
            provider["travel"] = {"D_T": 999}
            provider["type"] = "hxd"
            provider_list.append(provider)
        except:
            continue
    return provider_list

def get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj,pianyi_x_ave,pianyi_y_ave):
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    select_demands_sql = '''select keyid_popu,p041200,p041201,p041202,p041203,p041204,
                            p041205,p041206,p041207,p041208,p041209,p041210,p041211,p041212,
                            p041213,p041214,p041215,p041216,p041217,p041218,p041219,p041220,
                            p041221,p041222,p041223,x,y from POPU0412_JOIN_PARKS_JOIN_PS_PO order by id1 asc  '''
    # 执行
    cursor.execute(select_demands_sql)
    rs = cursor.fetchall()
    # demand_list=[]
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            population = list(row[1:25])
            x = row[25]+pianyi_x_ave
            y = row[26]+pianyi_y_ave
            demand = {}
            demand["demand_id"] = key_id
            demand["x"] = x
            demand["y"] = y
            demand["populist"] = population
            demand["provider"] = provider_list
            mongo_operater_obj.insert_record(demand)
        except:
            continue


if __name__=="__main__":
    # 计算可视化图和高德导航之间的偏移量
    # 鼓楼区吾悦广场停车场：高德坐标：118.776177,32.067054， ARCGIS坐标：118.769723  32.069206
    # 鼓楼区花卉文化市停车场：高德坐标：118.738079,32.062157，ArcGIS坐标：118.731473  32.064196
    # 溧水区溧水区农副产品交易中心停车场：高德坐标：119.016072,31.650429，ARCGIS坐标：119.009363  31.652310
    pianyi_x_1=118.776177-118.769723
    pianyi_y_1=32.067054-32.069206
    pianyi_x_2=118.738079-118.731473
    pianyi_y_2=32.062157-32.064196
    pianyi_x_3=119.016072-119.009363
    pianyi_y_3=31.650429- 31.652310
    print(pianyi_x_1, pianyi_y_1)
    print(pianyi_x_2, pianyi_y_2)
    print(pianyi_x_3, pianyi_y_3)
    pianyi_x_ave=(pianyi_x_1+pianyi_x_2+pianyi_x_3)/3
    pianyi_y_ave=(pianyi_y_1+pianyi_y_2+pianyi_y_3)/3
    print(pianyi_x_ave, pianyi_y_ave)
    #0.006589666666670269 -0.0020240000000022462
    # 说明高德坐标X方向上普标偏大，而Y方向上普遍偏小，接下来都加上这个偏移量即可

    provider_list = initial_providers(pianyi_x_ave,pianyi_y_ave)
    # mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_theory")
    # mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_jy_time")
    mongo_operater_obj = a_mongo_operater_PS_NJALL_TIME.MongoOperater("admin", "moo_ps_njall_time")
    get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj,pianyi_x_ave,pianyi_y_ave)

