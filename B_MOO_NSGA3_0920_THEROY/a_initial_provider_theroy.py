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
    # select_providers_sql = '''select key_id,x ,y,PS from T55_provider_points t order by PS DESC'''
    select_providers_sql = '''select keyid,x ,y,PS from T1919_provider_points t order by PS DESC'''
    # 执行
    cursor.execute(select_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]
            y = row[2]
            ps = row[3]

            provider={}
            provider["provider_id"]=key_id
            provider["x"]=x
            provider["y"]=y
            provider["ps"]=ps
            provider["travel"]={"D_T":999}
            provider_list.append(provider)
        except:
            continue
    return provider_list

def get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj):
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    # select_demands_sql = '''select KEY_ID,pdd,x,y from T55_demand_points'''
    select_demands_sql = '''select KEYID,pdd,x,y from T1919_demand_points'''
    # 执行
    cursor.execute(select_demands_sql)
    rs = cursor.fetchall()
    # demand_list=[]
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            population = row[1]
            x = row[2]
            y = row[3]
            demand = {}
            if key_id==None:
                print("test")
            demand["demand_id"] = key_id
            demand["x"] = x
            demand["y"] = y
            demand["population"] = population
            demand["provider"]= caculate_distance(provider_list,x,y)
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
    mongo_operater_obj = a_mongo_operater_theory.MongoOperater("admin", "moo_ps_theory19")
    get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj)

