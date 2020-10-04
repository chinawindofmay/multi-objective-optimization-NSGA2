# encoding:utf-8
import arcpy
from arcpy import env
import arcpy.cartography as CA
import numpy  as  np




if __name__=="__main__":
    dets = np.loadtxt('./frequency_result.txt', delimiter=',')
    # env.workspace = r'Database Connections/POWERSTATION.sde'
    # env.workspace = r'C:/Users/Administrator/AppData/Roaming/ESRI/Desktop10.7/ArcCatalog/41powerstation.sde'
    arcpy.env.workspace = "E:\\02research\\002data\\powerstation\\理论测试数据"
    shp_file_name="T55_provider_points.shp"
    cursor = arcpy.UpdateCursor(shp_file_name)  # SearchCursor是方法，返回Cursor的实例对象
    for row in cursor:
        row.setValue("frequency", 0)
        cursor.updateRow(row)
        print("clear Ok")
    cursor = arcpy.UpdateCursor(shp_file_name)  # SearchCursor是方法，返回Cursor的实例对象
    for row in cursor:
        keyid= row.getValue("key_id")
        frequency=row.getValue("frequency")
        for i in range(dets.shape[0]):
            for j in range(dets.shape[1]-1):
                if dets[i,j]==keyid:
                    frequency+=dets[i,-1]
        row.setValue("frequency", frequency)
        cursor.updateRow(row)
        print("update Ok")

