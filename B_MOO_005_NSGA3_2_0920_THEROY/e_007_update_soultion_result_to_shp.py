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
import numpy as np

import zlib

def update_solution_to_shp_layer():
    log_file_path = "./log"
    PROVIDERS_COUNT = 49
    D1_npzfile = np.load(log_file_path + "/jianye_solutions_300.npz")
    population_front0=D1_npzfile["population_front0"]
    best_y=D1_npzfile["objectives_fitness"]

    conn = cx_Oracle.connect('powerstation/powerstation@localhost/ORCL')
    cursor = conn.cursor()

    provider_id_list = [135584, 134958, 36828, 36827, 12768, 126268, 137795, 17725, 136506, 136213, 137153, 112916, 97916, 137494, 136205, 95001, 17086, 9047, 17085, 136200, 126265, 135894, 34729, 270109, 34674, 79487, 137151, 286491, 137493, 277418, 332880, 333171, 525836, 62210, 130473, 654155, 135282, 297573, 600875, 271352, 37295, 687066, 19947, 450590, 137800, 7521, 23163, 17363, 23162]

    for column in range(population_front0.shape[1]):
        update_sql = "update park_jy_test set solution1={0},solution2={1},solution3={2},solution4={3},solution5={4},solution6={5},solution7={6},solution8={7},solution9={8},solution10={9}  where KEY_ID={10}".format(
            population_front0[0, column], population_front0[1, column], population_front0[2, column], population_front0[3, column],
            population_front0[4, column], population_front0[5, column], population_front0[6, column],population_front0[7, column],
            population_front0[8, column], population_front0[9, column],provider_id_list[column])
        cursor.execute(update_sql)
        conn.commit()
    conn.close()
    print("字段更新成功")

if __name__=="__main__":
    update_solution_to_shp_layer()