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

    provider_id_list = [2347, 7518, 7519, 9047, 10490, 21502, 24545, 39111, 78875, 82610, 82612, 82871, 82902, 87090, 87644, 87667, 94732, 95019, 95020, 95278, 95284, 95296, 95587, 95593, 104420, 104422, 104703, 104992, 105010, 105015, 105020, 105303, 105306, 105318, 105587, 105604, 105607, 105608, 105875, 105878, 105893, 105899, 106138, 108742, 108755, 109029, 109039, 109316, 110276, 110603, 112301, 112328, 112623, 112626, 112644, 112926, 113197, 113200, 113202, 113204, 113205, 113217, 113221, 113222, 113487, 113492, 113496, 113500, 113509, 113514, 113759, 113760, 113765, 113767, 126529, 126533, 126538, 126539, 126551, 126553, 126812, 126815, 126823, 126825, 126837, 127975, 127977, 128271, 134666, 134943, 134946, 134947, 134959, 134963, 134966, 135266, 135271, 135272, 135277, 135284, 135285, 135588, 135591, 135882, 135888, 136202, 136203, 136209, 136212, 136483, 136485, 136489, 136504, 136827, 136846, 137147, 137150, 137153, 137154, 137177, 137487, 137778, 137782, 141271, 229465, 250033, 275888, 281677, 286774, 291323, 292890, 293930, 304353, 314529, 314581, 328935, 333170, 340693, 437232, 507674, 512590, 518370, 563510, 686145, 687282, 728660]

    for column in range(population_front0.shape[1]):
        update_sql = "update Parks_10_percentage set solution1={0},solution2={1},solution3={2},solution4={3},solution5={4},solution6={5},solution7={6},solution8={7},solution9={8},solution10={9}  where KEY_ID={10}".format(
            population_front0[0, column], population_front0[1, column], population_front0[2, column], population_front0[3, column],
            population_front0[4, column], population_front0[5, column], population_front0[6, column],population_front0[7, column],
            population_front0[8, column], population_front0[9, column],provider_id_list[column])
        cursor.execute(update_sql)
        conn.commit()
    conn.close()
    print("字段更新成功")

if __name__=="__main__":
    update_solution_to_shp_layer()