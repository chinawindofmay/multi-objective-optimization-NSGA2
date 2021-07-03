# encoding: utf-8
import pandas as pd
import arcpy


def create_line_method_1():
    TIME_SLOTS=["D_T08","D_T13","D_T18","D_T22"]
    for time_slot in TIME_SLOTS:
        print "正在处理{0}请等待".format(time_slot)
        ODs = pd.read_csv("./log/tzzs_data_{0}.csv".format(time_slot))
        fc=arcpy.CreateFeatureclass_management("./log","travel_time_{0}".format(time_slot),"POLYLINE")
        arcpy.AddField_management(fc,"time","FLOAT")
        cursor=arcpy.InsertCursor(fc)
        for index,row in ODs.iterrows():
            feature=cursor.newRow()
            point_array=arcpy.Array()
            origin_point=arcpy.Point(row[1],row[2])
            destination_point=arcpy.Point(row[3],row[4])
            point_array.add(origin_point)
            point_array.add(destination_point)
            point_line=arcpy.Polyline(point_array)
            feature.shape=point_line
            feature.time=row[5]
            cursor.insertRow(feature)
        print "处理完毕"

def create_line_method_2():
    # Set local variables
    input_table = "E:\\learning resources\\python\\OD\\tzzs_data_D_T08.csv"
    out_lines = "E:\\learning resources\\python\\OD\\T08XYtoline"
    # XY To Line
    arcpy.XYToLine_management(input_table, out_lines,
                              "dx", "dy", "px",
                              "py", "GEODESIC", "t")

if __name__=="__main__":
    create_line_method_1()