#encoding:utf-8
import urllib.request		# 导包时可以在上一个包后面加问号，再加上另一个包
import re
import os
import pandas as pd


# 遍历文件夹
def walkFile(file):
    files_path=[]
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            print(os.path.join(root, f))
            files_path.append(os.path.join(root, f))
        # 遍历所有的文件夹
        #for d in dirs:
            #print(os.path.join(root, d))
    return files_path

def search_from_baidu_by_name(person_name):
    result_str=[]
    keywd="{0}%20南京师范大学".format(person_name)
    keywd=urllib.request.quote(keywd)	# 注意转码，浏览器的网址框不接受中文字符，需要转码处理但是Python不会自动帮我们转码，所以我们这调用quote()函数来转码
    for i in range(1,3):				# 这里循环是因为要使用前2页的数据
        url="http://www.baidu.com/s?wd="+keywd+"&pn="+str((i-1)*10)	# 拼接成链接，格式的话需要自己提前查看，找到规律然后再拼接出来
        data=urllib.request.urlopen(url).read().decode("utf-8")		# 读取链接到的HTML文件
        pat="title:'(.*?)',"
        pat2='"title":"(.*?)",'
        rst1=re.compile(pat).findall(data)		# 使用正则来匹配标题，注意：这里有两个匹配是因为匹配时，获取标题的格式可能不同，这里有两个，所以写两种，但其实是可以综合在一起的
        rst2=re.compile(pat2).findall(data)
        for j in range(0,len(rst1)):	# 输出数据
            # print(rst1[j])
            result_str.append(rst1[j])
        for z in range(0,len(rst2)):
            # print(rst2[z])
            result_str.append(rst2[z])
    return result_str


def single_file_srapy():
    files_path = "D:/01-研究稿件/049-南师大地科院名人信息爬取/地理科学学院历届毕业生名单/地科院历届毕业生汇总大名单.xlsx"
    df = pd.read_excel(files_path)
    height, width = df.shape
    print(height, width, type(df))
    #     print(df)
    if "姓名" in list(df.columns.values):
        names = df["姓名"]
        result_str_list = []
        try:
            for index, value_name in names.items():
                result_str = search_from_baidu_by_name(person_name=value_name)
                result_str_list.append('#'.join(result_str))
                print(index,"--OK ")
            df["百度搜索结果"] = pd.Series(result_str_list)
            df.to_excel(files_path[:-5] + '结果.xls')
        except:
            print("error")




def mulite_file_srapy():
    files_path = walkFile("D:/01-研究稿件/049-南师大地科院名人信息爬取/地理科学学院历届毕业生名单")
    for item in files_path:
        df = pd.read_excel(item)
        height, width = df.shape
        print(height, width, type(df))
        #     print(df)
        if "姓名" in list(df.columns.values):
            names = df["姓名"]
            result_str_list = []
            try:
                for index, value_name in names.items():
                    result_str = search_from_baidu_by_name(person_name=value_name)
                    result_str_list.append('#'.join(result_str))
                df["百度搜索结果"] = pd.Series(result_str_list)
                df.to_excel(item[:-5] + '结果.xls')
            except:
                print("error")


if  __name__=="__main__":
    single_file_srapy()