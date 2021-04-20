#coding:utf-8
import sys
import math
import numpy as np
import cx_Oracle
import json
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks",context="notebook")
import jenkspy
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

def show_graph(CVs_col):
    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.xlabel('Time slots', font2)
    plt.ylabel('CV value', font2)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    ax = plt.gca();  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);  ####设置上部坐标轴的粗细

    x = np.linspace(0, len(CVs_col), len(CVs_col))
    plt.plot(x, CVs_col, 'g')
    # axis.plot(Y, 'g.')  # 每次循环的目标函数值
    plt.show()


def update_cv_value_to_tabel(CVs_rows,IDs):
    for i in range(END_ROWS):
        update_sql = "update POLYGON_20190412 set cv={0}  where id1={1}".format(CVs_rows[i, 0], IDs[i, 0])
        cursor.execute(update_sql)
        conn.commit()
    conn.close()
    print("字段更新成功")


if __name__=="__main__":
    END_ROWS=873
    conn = cx_Oracle.connect('EMS/EMS@localhost/ORCL')
    cursor = conn.cursor()
    commit = 0
    select_popu_sql = '''select  P041200,P041201,P041202,P041203,
       P041204,P041205,P041206,P041207,
       P041208,P041209,P041210,P041211,
       P041212,P041213,P041214,P041215,
       P041216,P041217,P041218,P041219,
       P041220,P041221,P041222,P041223,ID1
        from POLYGON_20190412 t'''
    cursor.execute(select_popu_sql)
    rs = cursor.fetchmany(END_ROWS)
    pop=np.full(shape=(END_ROWS,24),fill_value=0.0)
    IDs=np.full(shape=(END_ROWS,1),fill_value=0)
    CVs_rows=np.full(shape=(END_ROWS, 1), fill_value=0.0)
    CVs_cols=np.full(shape=(24), fill_value=0.0)
    for i, row in enumerate(rs):
        IDs[i,0]=row[24]
        for j in range(24):
            pop[i,j]=row[j]

    for i in range(END_ROWS):
        std=np.std(pop[i,:])
        mean=np.mean(pop[i,:])
        if mean==0.0:
            mean=0.0001
        cv_row= std / mean
        CVs_rows[i, 0]=cv_row

    for k in range(24):
        std=np.std(pop[:,k])
        mean=np.mean(pop[:,k])
        if mean==0.0:
            mean=0.0001
        cv_col=std/mean
        CVs_cols[k]=cv_col

    update_cv_value_to_tabel(CVs_rows,IDs)
    show_graph(CVs_cols)

