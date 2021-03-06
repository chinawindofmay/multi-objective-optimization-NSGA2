# -*- coding:utf-8 -*-

"""
用于筛选出候选停车点
"""

import math
import json
#ORACLE库
import cx_Oracle
import pandas as pd
import os


conn_string='powerstation/powerstation@192.168.90.2/ORCL'

def initial_parks():
    provider_list=[]
    conn = cx_Oracle.connect(conn_string)
    cursor = conn.cursor()
    # 第一步:已经建成的，不做调整
    select_providers_sql = '''select objectid,tt.adname||tt.name as name,tt.locationx-0.00137,tt.locationy,tt.type,tt.adname,tt.address  from ( 
select objectid,name,type,locationx,locationy,t.adname,t.address  from NJPOI2019 t where type like '%停车%' and type not like '%出入口' and name not like '%小区%' and type not like '%专用%' and t.name not like '%出口%' and t.name not like '%入口%' order by adcode,name  asc
) tt'''
    # 执行
    cursor.execute(select_providers_sql)
    rs = cursor.fetchall()
    operation_list=[]
    result_df=pd.DataFrame(columns=('keyid','name','x','y',"type","distict","address"))
    k=0
    for i, row in enumerate(rs):
        print(i)
        try:
            objectid = row[0]
            name = row[1]
            x = row[2]
            y = row[3]
            typeob = row[4]
            distict=row[5]
            address=row[6]
            provider = {}
            provider["keyid"] = objectid
            provider["name"] = name
            provider["x"] = x
            provider["y"] = y
            provider["type"] = typeob
            provider["distict"] = distict
            provider["address"] = address
            if len(operation_list)==0:
                operation_list.append(provider)
            else:
                # 如果两类名字一样
                if operation_list[0]["name"]==name:
                    if math.sqrt((operation_list[0]["x"]-x)**2+(operation_list[0]["y"]-y)**2)*111.3<0.5:
                        # 距离也很近，可以删除一个
                        continue
                    else:
                        # 距离很远，可以保留
                        result_df.loc[k] = provider
                        k += 1
                else:
                    result_df.loc[k] = operation_list.pop(0)
                    k += 1
                    operation_list.append(provider)
        except:
            continue
    return result_df


if __name__=="__main__":
    result_df = initial_parks()
    filepath="./data/nj_public_parks.xls"
    if os.path.isfile(filepath):
        os.remove(filepath)
    result_df.to_excel(filepath)
    print("OK")


