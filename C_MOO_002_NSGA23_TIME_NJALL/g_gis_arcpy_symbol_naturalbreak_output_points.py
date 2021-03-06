#coding:utf-8

import arcpy
import jenkspy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# python2.\pip2.exe
# install.\jenkspy - 0.1
# .1 - cp27 - cp27m - win32.whl

def show_hist_of_breaks(data,break_values):
    '''
    可视化原数据与断点
    :param data: 原数据
    :param break_values: 断点
    :return:
    '''
    # 正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    n,_,_=plt.hist(data,bins=50,facecolor="DodgerBlue")
    max_value=max(n)
    for i in break_values:
        if i:
            plt.axvline(x=i,ls="--", c="gray",ymax=200)  # 添加垂直直线
            plt.text(i+800,max_value-200,str(i),rotation=90,fontsize=15)
        else:
            continue
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(u"区间",fontsize=18)
    plt.ylabel(u"频数/频率",fontsize=18)
    plt.ylim(0, max_value+10)
    plt.savefig('result.png',dpi=300)  # 保存图片
    plt.show()


def get_natural_breaks(data_list):
    '''
    自然间断点分级法，获取列表格式的断点
    :param data_list: 数据
    :return:断点值和断点标签
    '''
    breaks = jenkspy.jenks_breaks(data_list, nb_class=nb_class)
    print(breaks)
    dd=defaultdict(list)
    for k,va in [(v,i) for i,v in enumerate(breaks)]:
        dd[k].append(va)
    for i,v in enumerate(dd):
        if(len(dd[v])>1):
            breaks[dd[v][len(dd[v])-1]]+=0.1
    print(breaks)
    breaks_values = []
    breaks_labels = []
    for i, item in enumerate(breaks[0:-1]):
        # breaks_values.append(round(item / round_num) * round_num)
        # if i < len(breaks) - 1:
        #     breaks_labels.append(str(round(item / round_num) * round_num) + '-' + str(
        #         round(breaks[i + 1] / round_num) * round_num))
        # else:
        #     break
        breaks_labels.append(str(item)+'-'+str(breaks[i+1]))
    print(breaks_values)
    print(breaks_labels)
    return breaks, breaks_labels
    # dic_breaks_values = {}.fromkeys(breaks_values)  # 使用fromkeys方法创建一个字典，字典的键会自动去重
    # if (len(dic_breaks_values) == len(breaks_values)):
    #     print("————————分组完毕！继续执行———————")
    #     print(breaks_values)
    #     print(breaks_labels)
    #     return breaks_values, breaks_labels
    # else:
    #     print("请调整round_num或者nb_class参数的设置！！！！！")
    #     return


def output_img(mxd,lyr,breaks_values,breaks_labels):
    '''
    导出图片
    :param mxd: 地图文档
    :param lyr: 图层
    :param breaks_values: 断点值
    :param breaks_labels: 断点标签
    :return:
    '''
    if hasattr(lyr, 'symbologyType'):
        print(lyr.symbologyType)
        if lyr.symbologyType == 'GRADUATED_SYMBOLS':
            for index, field in enumerate(field_name):
                print("————————正在输出，第" + str(index+1) + "张地图——————————")
                lyr.symbology.valueField = field
                lyr.symbology.classBreakValues = breaks_values
                lyr.symbology.classBreakLabels = breaks_labels
                arcpy.RefreshActiveView()
                arcpy.RefreshTOC()
                # arcpy.mapping.ExportToJPEG(mxd,
                #                            file_path+"\\" + field + ".jpg",
                #                            data_frame=arcpy.mapping.ListDataFrames(mxd, "图层")[0],
                #                            df_export_width=1200,
                #                            df_export_height=1500,
                #                            resolution=600)
                arcpy.mapping.ExportToJPEG(mxd,file_path + "\\" + field + ".jpg")
        else:
            print("请将图层属性设置为“分级颜色”！！！")

def excute():
    '''
    运行
    :return:
    '''
    mxd = arcpy.mapping.MapDocument(mxd_file_path)
    for lyr in arcpy.mapping.ListLayers(mxd):
        if (lyr.name == lyr_name):
            with arcpy.da.SearchCursor(lyr, field_name) as cursor:
                data = []
                for row in cursor:  # 返回结果：row元组
                    row_list = list(row)
                    for i in range(len(row_list)):
                        if row_list[i]!=None:  #配置的区域限于现在已经有停车场的区域
                            data.append(row_list[i])
                # 增加代码，弥补最后处容易无法统计进去   20201006 周鑫鑫补充
                # data_np=np.array(data)
                # print(len(data_np))
                # print(np.argwhere(data_np == np.max(data_np)))
                # data_np[np.argwhere(data_np==np.max(data_np))]+=np.float64(0.1)
                # data=list(data_np)
                print(data)
                # driving time 200
                # cover people 200
                #breaks_values, breaks_labels=get_natural_breaks(data)
                breaks_values=[0.0, 1.0, 40.0, 80.0, 120.0,160.0, 200.0]
                breaks_labels=['0', '1-40','41-80', '81-120', '121-160', '160-200']

                # # dynamic and static
                # # dynamic
                # breaks_values = [0, 20, 30, 50, 70, 90, 100]
                # breaks_labels = ['0', '20-30', '30-50', '50-70', '70-90', '90-100']

                output_img(mxd,lyr,breaks_values,breaks_labels)
                data1=[]
                for item in data:
                    if item>0:
                        data1.append(item)
                show_hist_of_breaks(data1,breaks_values)
    print("over")

if __name__=="__main__":
    '''
    time:2020/08/17
    author:crazy

    自然间断点算法 https://blog.csdn.net/allenlu2008/article/details/103884170

    工具说明：
        数据：一个图层中多个字段；
        断点：主要针对整数；
        图层：符号设置为“分级符号”
    '''
    #
    # file_path="E:\\02research\\002data\\sjxl\\NanjingGrid"#文件路径
    # mxd_file_path="{0}\\popdata.mxd".format(file_path) #mxd名称
    # lyr_name=u"nj250_hadpop"#图层名称
    # field_name=["populati_1","pop08","pop133","pop18","pop22"]#字段名称

    # file_path="E:\\02research\\002data\\STA-20201006"#文件路径
    # mxd_file_path="{0}\\02model_population.mxd".format(file_path) #mxd名称
    # lyr_name=u"population"#图层名称
    # field_name=["pop08","pop13","pop18","pop22"]#字段名称

    # # driving time
    # file_path = "E:\\02research\\002data\\powerstation_nj_2021"  # 文件路径
    # mxd_file_path = "{0}\\powerstation_njall_2021_edit.mxd".format(file_path)  # mxd名称
    # lyr_name = u"新建充电桩"  # 图层名称
    # field_name = ["solution1", "solution2", "solution3", "solution4", "solution5",
    #               "solution6", "solution7", "solution8"]  # 字段名称

    # cover people
    file_path = "E:\\02research\\002data\\powerstation_nj_2021"  # 文件路径
    mxd_file_path = "{0}\\powerstation_njall_2021_coverpeople.mxd".format(file_path)  # mxd名称
    lyr_name = u"新建充电桩"  # 图层名称
    field_name = ["solution1", "solution2", "solution3"]  # 字段名称

    # file_path="E:\\02research\\002data\\powerstation_nj_2021"#文件路径
    # mxd_file_path="{0}\\powerstation_njall_2021_static.mxd".format(file_path) #mxd名称
    # mxd_file_path="{0}\\powerstation_njall_2021_dynamic.mxd".format(file_path) #mxd名称
    # lyr_name=u"新建充电桩"#图层名称
    # field_name=["solution1","solution2"]#字段名称

    nb_class = 6  # 级数
    round_num = 1.0  # 取整

    excute()









