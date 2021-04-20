#coding:utf-8
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import cx_Oracle

#从数据库中读取数据
def read_oracle_data(field_name):
    ORACLE_CONN_STR = cx_Oracle.makedsn('localhost', '1521', 'orcl')
    ORACLE_USER = 'powerstation'
    ORACLE_PASSWORD = 'powerstation'
    conn = cx_Oracle.connect(dsn=ORACLE_CONN_STR, user=ORACLE_USER, password=ORACLE_PASSWORD)
    cursor = conn.cursor()
    sql = '''
          select {0} from popu0412_join_parks_join_ps t
          '''.format(field_name)
    cursor.execute(sql)
    # 构建每条记录的请求体
    rows = cursor.fetchall()
    n_fetched = len(rows)
    data=[]
    for row in rows:
        data.append(row[0])
    cursor.close()
    conn.close()
    return data

#绘制统计图
def show_hist_graph(data,moment,bin_count,x_lim,y_ticks,y_accumlate_percent_tricks,pie_breaks,pie_labels):
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size']=30
    fig = plt.figure()
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(8, 6)
    ax = fig.add_subplot(1,1,1)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')

    #hist graph
    ax1=fig.add_subplot(111)
    ax1.set_xlim(x_lim)
    rows=len(data)
    n,bins,patches=ax1.hist(data, bins=bin_count, normed=0, facecolor="dodgerblue", edgecolor="dimgray", alpha=0.7,label='Accessibility frequency')
    plt.xlabel("Accessibility value at {0}".format(moment))
    plt.ylabel("Accessibility frequency")
    plt.yticks(y_ticks)
    ax2=ax1.twinx()
    n2,bins2,patches2=plt.hist(data, bins=bin_count,density=True, histtype='step', cumulative=True,alpha=0.0)

    #累积曲线，分布函数：sigma标准差，mu均值
    mu=0
    sigma=0
    sig=0
    sum=0.0
    for i in range(0,rows):
        sum+=data[i]
    mu=sum/(rows)
    for j in range(0,rows):
        sig += (data[j]-mu)*(data[j]-mu)
    sigma=math.sqrt(sig/(rows))
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins2 - mu))**2))
    y = y.cumsum()
    y /= y[-1]
    ax2.plot(bins2, y, 'k--', linewidth=1.5, label='Cumulative')
    plt.ylabel("Cumulative percent")
    plt.yticks(y_accumlate_percent_tricks)
    fig.legend(loc='upper left',bbox_to_anchor=(0.55, 0.85),ncol=2,fontsize=16)#双y轴合并图例

    #圆饼图
    colors = ['coral', 'darkturquoise', 'limegreen', 'orchid']
    a = plt.axes([.46, .16, .29, .58], facecolor='k')
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    sizes = []
    for i in range(0, bin_count):
        if bins[i] < pie_breaks[0]:
            s1+=n[i]
        if (bins[i] < pie_breaks[1]) & (bins[i] >= pie_breaks[0]):
            s2 += n[i]
        if (bins[i] < pie_breaks[2]) & (bins[i] >= pie_breaks[1]):
            s3 += n[i]
        if (bins[i]>pie_breaks[2]):
            s4 += n[i]
    sizes = [s1, s2, s3, s4]  # 饼图中的数据
    explode = (0,0,0,0)
    wedges, texts, autotexts=plt.pie(sizes,explode=explode,autopct='%1.3f%%',shadow=False,startangle=180,colors=colors)
    ax.legend(wedges, pie_labels,
              loc="center right",
              bbox_to_anchor=(0.77,-0.18, 1., 1),
              fontsize=12
              )
    plt.show()
    # fig.savefig('./log/{0}.jpg'.format(moment), dpi=200,bbox_inches = 'tight')  # 设置保存图片的分辨率

if __name__=="__main__":
    data=read_oracle_data("access22")
    moment="22:00"
    x_lim=[0,0.2]
    bin_count=200
    y_ticks=range(0,30,5)
    y_accumlate_percent_tricks=[0.25,0.50,0.75,1.00]
    pie_breaks=[0.05,0.10,0.15]  # 三个点，分四段
    pie_labels = ["0.00-0.05", "0.05-0.10", "0.10-0.15", ">0.15"]  #四段
    show_hist_graph(data,moment,bin_count,x_lim,y_ticks,y_accumlate_percent_tricks,pie_breaks,pie_labels)
