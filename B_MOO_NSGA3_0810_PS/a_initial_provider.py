# -*- coding:utf-8 -*-

"""
用于初始化provider对象
"""

import math
import json
#ORACLE库
import cx_Oracle
from pymongo import MongoClient
import a_mongo_operater

import zlib

conn_string='powerstation/powerstation@192.168.60.41/ORCL'

def initial_providers():
    provider_list=[]
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    select_providers_sql = '''select key_id,经度 as x ,纬度 as y,快充桩 as quickcharge ,慢充桩 as slowcharge,地址 as address from PS_JY t'''
    # 执行
    cursor.execute(select_providers_sql)
    rs = cursor.fetchall()
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            x = row[1]
            y = row[2]
            quickcharge = row[3]
            slowcharge = row[4]
            address=row[5]

            provider={}
            provider["provider_id"]=key_id
            provider["x"]=x
            provider["y"]=y
            provider["quickcharge"]=quickcharge
            provider["slowcharge"]=slowcharge
            provider["address"]=address
            provider["travel"]={"D_T":999}
            provider_list.append(provider)
        except:
            continue
    return provider_list

def get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj):
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    select_demands_sql = '''select KEY_ID,名称 as name,population,x,y from village_JY'''
    # 执行
    cursor.execute(select_demands_sql)
    rs = cursor.fetchall()
    # demand_list=[]
    for i, row in enumerate(rs):
        try:
            key_id = row[0]
            name = row[1]
            population = row[2]
            x = row[3]
            y = row[4]
            demand = {}
            demand["demand_id"] = key_id
            demand["name"] = name
            demand["x"] = x
            demand["y"] = y
            demand["population"] = population
            demand["provider"]=provider_list
            mongo_operater_obj.insert_record(demand)
        except:
            continue


if __name__=="__main__":
    provider_list = initial_providers()
    mongo_operater_obj = mongo_operater.MongoOperater("admin", "moo_ps")
    get_all_demonds_and_save_in_mongodb(provider_list,mongo_operater_obj)

