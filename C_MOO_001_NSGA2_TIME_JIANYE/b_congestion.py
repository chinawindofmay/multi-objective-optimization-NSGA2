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

def calculate_change_value(array_mtm):
    if np.all(array_mtm < 100):
        sum_value=0
        count=0
        for item in array_mtm[0:-1]:
            if item >array_mtm[-1]:
                sum_value+=(item-array_mtm[-1])/array_mtm[-1]
                count+=1
        return 0 if count==0 else sum_value/count
    else:
        return 0

def show_congestion_line_chart(PROVIDER_FIELD_LIST):
    cursor = conn.cursor()
    commit = 0
    select_providers_sql = '''select objectid,{0},{1},{2},{3} from {4}'''.format(PROVIDER_FIELD_LIST[0],PROVIDER_FIELD_LIST[1],PROVIDER_FIELD_LIST[2],PROVIDER_FIELD_LIST[3],SHP_LAYER_NAME)
    cursor.execute(select_providers_sql)
    rs = cursor.fetchmany(END_ROWS)
    congistion_PT_list=[]
    congistion_D_list=[]
    congistion_W_list=[]
    congistion_B_list=[]
    for i, row in enumerate(rs):
        objectid = row[0]
        print(objectid)
        provider08 = row[1].read()
        provider13 = row[2].read()
        provider18 = row[3].read()
        provider22 = row[4].read()
        provide08_json_list = json.loads(provider08)
        provide13_json_list = json.loads(provider13)
        provide18_json_list = json.loads(provider18)
        provide22_json_list = json.loads(provider22)
        for j in range(26):
            pt_array_PT = np.array([provide08_json_list[j]["travel"]["TP_T"],
                                 provide13_json_list[j]["travel"]["TP_T"],
                                 provide18_json_list[j]["travel"]["TP_T"],
                                 provide22_json_list[j]["travel"]["TP_T"]])
            congistion_PT = calculate_change_value(pt_array_PT)
            pt_array_D = np.array([provide08_json_list[j]["travel"]["D_T"],
                                    provide13_json_list[j]["travel"]["D_T"],
                                    provide18_json_list[j]["travel"]["D_T"],
                                    provide22_json_list[j]["travel"]["D_T"]])
            congistion_D = calculate_change_value(pt_array_D)
            pt_array_B = np.array([provide08_json_list[j]["travel"]["B_T"],
                                   provide13_json_list[j]["travel"]["B_T"],
                                   provide18_json_list[j]["travel"]["B_T"],
                                   provide22_json_list[j]["travel"]["B_T"]])
            congistion_B = calculate_change_value(pt_array_B)
            pt_array_W = np.array([provide08_json_list[j]["travel"]["W_T"],
                                   provide13_json_list[j]["travel"]["W_T"],
                                   provide18_json_list[j]["travel"]["W_T"],
                                   provide22_json_list[j]["travel"]["W_T"]])
            congistion_W = calculate_change_value(pt_array_W)
            congistion_PT_list.append(congistion_PT)
            if congistion_PT>1:
                print("test")
            congistion_D_list.append(congistion_D)
            congistion_W_list.append(congistion_W)
            congistion_B_list.append(congistion_B)

    # x_data = [i for i in range(len(congistion_PT_list))]
    # plt.plot(x_data, congistion_PT_list, color='blue', linewidth=2.0, linestyle='--')
    # plt.plot(x_data, congistion_D_list, color='red', linewidth=3.0, linestyle='-.')
    # plt.plot(x_data, congistion_W_list, color='green', linewidth=2.0, linestyle='-.')
    # plt.plot(x_data, congistion_B_list, color='grey', linewidth=2.0, linestyle='-.')
    # plt.show()

    np_array=np.hstack((np.array(congistion_PT_list).T.reshape((len(congistion_PT_list),1)),
                       np.array(congistion_D_list).T.reshape((len(congistion_PT_list),1)),
                       np.array(congistion_B_list).T.reshape((len(congistion_PT_list),1)),
                       np.array(congistion_W_list).T.reshape((len(congistion_PT_list),1)))
                       )
    result_np_array=np_array[np_array[:,1]>0,:]
    # # st_rt_np_ay=np.sort(a=result_np_array,axis=0,order=0)   所有列一起排序
    # result_np_array = result_np_array[np.argsort(result_np_array[:, 0])]   #按某一列排序

    # breaks_PT = jenkspy.jenks_breaks(congistion_PT_list, nb_class=4)
    # print(breaks_PT)
    # breaks_D = jenkspy.jenks_breaks(congistion_D_list, nb_class=4)
    # print(breaks_D)
    x_data = [i for i in range(result_np_array.shape[0])]
    plt.plot(x_data, result_np_array[:,0], color='blue', linewidth=1.0 ,alpha=0.9, linestyle='-.',label="Public transportation")
    plt.plot(x_data, result_np_array[:, 1], color='grey', linewidth=1.0, alpha=0.9, linestyle='-.',label="Driving")
    plt.plot(x_data, result_np_array[:,2], color='green', linewidth=2.0, linestyle='-.',label="Bicycling")
    plt.plot(x_data, result_np_array[:,3], color='red', linewidth=1.0, linestyle='-.',label="Walking")
    plt.tick_params(labelsize=15)  # 刻度字体大小13
    plt.ylabel("Traffic congestion coefficient", fontdict={'family': 'Times New Roman', 'size': 16,'fontweight':'bold'})
    plt.xlabel("OD flows series", fontdict={'family': 'Times New Roman', 'size': 16,'fontweight':'bold'})
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.savefig('./congestion.jpg', dpi=200)
    plt.show()