# -*- coding:utf-8 -*-

"""
用于初始化provider对象
"""

import math
import json
#ORACLE库
import cx_Oracle
from pymongo import MongoClient
import a_mongo_operater_theory
import copy

import zlib

conn_string='powerstation/powerstation@192.168.90.2/ORCL'

def initial_providers():
    provider_list=[]
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    # 第一步:已经建成的，不做调整
    select_providers_sql = '''select key_id,经度 as x ,纬度 as y,快充桩 as quickcharge , 地址 as address from PS_JY t order by key_id asc '''
    # 执行
    cursor.execute(select_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]
            y = row[2]
            quickcharge = row[3]
            address = row[4]

            provider={}
            provider["provider_id"]=key_id
            provider["x"]=x
            provider["y"]=y
            provider["quickcharge"]=quickcharge
            provider["address"]=address
            provider["travel"]={"D_T":999}
            provider_list.append(provider)
        except:
            continue

    # 第2步:未建的，要做增加
    select_petential_providers_sql = '''select key_id,locationx as x ,locationy as y,0 as quickcharge ,address from parks_10_percentage t order by key_id asc '''
    # 执行
    cursor.execute(select_petential_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]
            y = row[2]
            quickcharge = row[3]
            address = row[4]

            provider = {}
            provider["provider_id"] = key_id
            provider["x"] = x
            provider["y"] = y
            provider["quickcharge"] = quickcharge
            provider["address"] = address
            provider["travel"] = {"D_T": 999}
            provider_list.append(provider)
        except:
            continue
    return provider_list

def get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj):
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    select_demands_sql = '''select KEYID,pop041200,pop041201,pop041202,pop041203,pop041204,
                            pop041205,pop041206,pop041207,pop041208,pop041209,pop041210,pop041211,pop041212,
                            pop041213,pop041214,pop041215,pop041216,pop041217,pop041218,pop041219,pop041220,
                            pop041221,pop041222,pop041223,x,y from pop_fishnet1000_20190412 order by KEYID asc  '''
    # 执行
    cursor.execute(select_demands_sql)
    rs = cursor.fetchall()
    # demand_list=[]
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            population = list(row[1:25])
            x = row[25]
            y = row[26]
            demand = {}
            demand["demand_id"] = key_id
            demand["x"] = x
            demand["y"] = y
            demand["populist"] = population
            demand["provider"] = provider_list
            mongo_operater_obj.insert_record(demand)
        except:
            continue

def caculate_distance(provider_list,demand_x,demand_y):
    provider_list_copy=copy.deepcopy(provider_list)
    for i in range(len(provider_list_copy)):
        distance=((provider_list_copy[i]["x"]-demand_x)**2+(provider_list_copy[i]["y"]-demand_y)**2)**0.5
        provider_list_copy[i]["travel"]={"D_T":round(distance,3)}
    return provider_list_copy


if __name__=="__main__":
    provider_list = initial_providers()
    # mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_theory")
    # mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_jy_time")
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_jy_time1000")
    get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj)

