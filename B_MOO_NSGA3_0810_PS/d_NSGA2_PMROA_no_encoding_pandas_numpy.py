#coding:utf-8
"""
关于整个算法的基本说明：
算法功能：实现对可达性和公平性的帕累托解求解过程。
算法参照：
    # Program Name: NSGA-II.py
    # Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
    # Author: Haris Ali Khan
    # Supervisor: Prof. Manoj Kumar Tiwari
算法的改造过程：
    第一步：20200129-20200130，资料收集，NSGA基本原理再理解，调试原有的代码（发现没法验证对错）于是乎github上重新寻找代码
            理解基础的NSGA-II.py，单个x，两个简单数学函数目标，数学制图，详细见SGAII-traditional.py
    第二步：20200131，将NSGA-II.py算法改成求解x1,x2，三个数学曲面目标，数学制图，详细见NSGAII-math-valiation.py
            这一步非常重要，因为这一步决定了对算法理解和改造的基础正确性问题
            实现了三个曲面的求解，验证了准确性，且实现了制图表达，便于接下来的改造
    第三步：20200201-20200202，改成：26个x，可达性和公平性两个数学目标，需要修改initial_population、crossover、mutation、fitness、accessbility数据读取等
            过程中主要解决了几个问题：limitation改造问题；fitness计算慢的问题；去掉了编码和解码的过程（从而简化了问题的可读性）
            测试50种群，300代的结果
            将小数位结果整数化
            2030医生数量预测
    第四步：将原有基于Python list的存储和运算机制修改为基于pandas和Numpy的机制，从而增加运行效率
            2020年3月8日，完成了pandas和numpy的存储和修改，发现pandas效率为一次运行fitness时间在10S
            晚上，鼓起勇气，将其修改为numpy的，具体查看图片:d_numpy存储机制.jpg，一次运行fitness的时间在0.03S
    第五步：重新思考确立论文题目：高数量Or高质量 医疗资源优化配置帕累托解求解过程，将马太效应和分级医疗引入到本次的论文题目中
            阅读论文，增加第三个评价指标，
            目前需要做四件事情：整理代码并提交到Github、分级医疗的基础数据检查、Huff模型再审视、修改模型的代码
    第五步：思考变长NSGA;
    第六步：思考Q-NSGA2；
    第七步：思考其他的目标优化问题求解；
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from e_fitness_gravity_equality import *
from numpy import *
from sklearn import preprocessing
import a_mongo_operater
import time



"""
用于初始化Population
#softmax函数
"""
def softmax_function(t_list):
    try:
        z_exp = [math.exp(i) for i in t_list]
        sum_z_exp = sum(z_exp)
        softmax_list = [round(i / sum_z_exp, 3) for i in z_exp]
        return np.array(softmax_list)
    except:
        return

"""
生成随机分配概率List
"""
def softmax_random():
    x_list=np.array([random.random() for i in range(0, SOLUTION_LEN)])
    list_z_score = preprocessing.scale(1 / x_list)  # 这里1和10和100对其结果没有影响
    # print("z_score", list_z_score)
    softmax_probability = softmax_function(list_z_score)
    return softmax_probability

"""
名称：初始化种群函数
作用：初始化原始种群
"""
def initial_population():
    population = np.empty(shape=[POP_SIZE, SOLUTION_LEN],dtype=np.float32)
    for i in range(POP_SIZE):
        solution=np.array([round(LIMINTATION * probability, 3) for probability in softmax_random()],dtype=np.float32)
        population[i,:]=solution
    return population

"""
交叉算子，实现两个solution交叉
"""
def crossover_mutation_limitation(solution_a, solution_b):
    ##第1列和第2列存放概率，第3存放交换结果，第4列出发1或0，表示TRUE和FALSE，
    new_solution_sign = np.random.random((SOLUTION_LEN,4))
    #第一步交叉
    for i in range(0, SOLUTION_LEN):
        if new_solution_sign[i,0]<CROSSOVER_PROB_THRESHOLD:
            new_solution_sign[i,2]=solution_b[i]
            new_solution_sign[i,3]=1
        else:
            new_solution_sign[i, 2] = solution_a[i]
            new_solution_sign[i, 3] = 0
    # 第二步变异
    for i in range(0, SOLUTION_LEN):
        if new_solution_sign[i,1] < MUTATION_PROB__THRESHOLD:
            x = round(MIN_X + (MAX_X - MIN_X) * random.random()+0.1,3)
            new_solution_sign[i, 2] =x
            new_solution_sign[i, 3] =1

    #第三步 求解总值
    sum_x = np.sum(new_solution_sign[:,2])
    sum_no_adjust_x = 0
    for i in range(0, SOLUTION_LEN):
        if new_solution_sign[i][3] == 0:
            sum_no_adjust_x+=new_solution_sign[i,2]
    #第四步：完成交叉和变异之后，要对整体做一次Limitation
    adjust_parameter= (LIMINTATION - sum_no_adjust_x) / (sum_x - sum_no_adjust_x)
    new_solution = np.empty(shape=(SOLUTION_LEN,), dtype=np.float32)
    #adjust_parameter>1，则表示要对TRUE的x做放大操作，乘
    #adjust_parameter<1，则表示要对TRUE的x做缩小操作，乘
    for i in range(0, SOLUTION_LEN):
        if new_solution_sign[i,3]==1:
            #开始调整数值
            new_solution[i] = round(new_solution_sign[i,2]*adjust_parameter,3)
        else:
            new_solution[i] = new_solution_sign[i,2]
    return new_solution



"""
基于Numpy存储,计算适应度，包括了可达性和公平性两个子函数
"""
def fitness_numpy(demands_numpy, population,DEMANDS_COUNT,access_type):
    y1_values_double = np.empty(shape=(np.shape(population)[0],),dtype=np.float32)
    y2_values_double = np.empty(shape=(np.shape(population)[0],),dtype=np.float32)
    for i_solution in range(np.shape(population)[0]):
        #这一步计算出了当前solution下，每个医院的gravtiy值
        solution=population[i_solution,:]
        #开始时间
        # start_time = time.time()
        if access_type=="g":
            update_every_single_provider_gravity_value_numpy(demands_numpy, solution, DEMANDS_COUNT)
        # elif access_type=="h":
        #     update_every_single_hospital_h2sfca_value_numpy(demands_numpy, solution, DEMANDS_COUNT)
        # 可达性适应度值，该值越大越好
        y1_values_double[i_solution]=np.nansum(demands_numpy[:,0,0,1])   #获取到每一个的gravity value  之所以用nan，是为了过滤掉nan的值
        # 计算公平性数值，该值越小表示越公平
        # y2_values_double[i_solution]=np.cov(demands_numpy[:,0,0,1])  这是未考虑人口数影响的公平性方程
        y2_values_double[i_solution]=calculate_global_accessibility_equality_numpy(demands_numpy)
        # 结束时间
        # end_time1 = time.time()
        # print('calculate_gravity_value() Running time: %s Seconds' % (end_time1 - start_time))
    return y1_values_double, y2_values_double



"""
NSG2的基础组成部分
#Function to find index of list
"""
def get_index_of(a, list_obj):
    for i in range(0, len(list_obj)):
        if list_obj[i] == a:
            return i
    return -1

"""
NSG2的基础组成部分
#Function to sort by values
"""
def sort_by_values(front, y_values):
    sorted_list = []
    while(len(sorted_list)!=len(front)):
        if get_index_of(min(y_values), y_values) in front:
            sorted_list.append(get_index_of(min(y_values), y_values))
        y_values[get_index_of(min(y_values), y_values)] = math.inf
    return sorted_list

"""
NSG2的基础组成部分
#Function to carry out NSGA-II's fast non dominated sort
"""
def fast_non_dominated_sort(y1_values, y2_values):
    S=[[] for i in range(0, np.shape(y1_values)[0])]
    fronts = [[]]
    n=[0 for i in range(0, np.shape(y1_values)[0])]
    rank = [0 for i in range(0, np.shape(y1_values)[0])]

    for p in range(0, np.shape(y1_values)[0]):
        S[p]=[]
        n[p]=0
        for q in range(0, np.shape(y1_values)[0]):
            # 这是目标函数，y1求大值，y2求小值
            if (y1_values[p] > y1_values[q] and y2_values[p] < y2_values[q] )  :
                if q not in S[p]:
                    # 个体p的支配集合Sp计算
                    S[p].append(q)
            elif (y1_values[p] < y1_values[q] and y2_values[p] > y2_values[q] ) :
                # 被支配度Np计算
                # Np越大，则说明p个体越差
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in fronts[0]:
                fronts[0].append(p)

    i = 0
    while(fronts[i] != []):
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        fronts.append(Q)

    del fronts[len(fronts)-1]
    return fronts

"""
NSGA2的基础组成部分
#Function to calculate crowding distance
"""
def crowding_distance(y1_values, y2_values, front):
    distance = [0 for i in range(0,len(front))]
    #根据y1的值做一次排序
    sorted1 = sort_by_values(front, y1_values[:])
    #根据y2的值做一次排序
    sorted2 = sort_by_values(front, y2_values[:])
    #第一个个体和最后一个个体，定义为无限远
    distance[0] = DISTANCE_INFINTE
    distance[len(front) - 1] = DISTANCE_INFINTE
    #计算中间个体的距离
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted1[k + 1]] - y2_values[sorted1[k - 1]]) / (max(y1_values) - min(y1_values)+DELATE)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (y1_values[sorted2[k + 1]] - y2_values[sorted2[k - 1]]) / (max(y2_values) - min(y2_values)+DELATE)
    return distance

"""
#二维制图表达
"""
def draw_2d_plot(y1_values,y2_values):
    fig = plt.figure(figsize=(12, 12))
    ax11 = fig.add_subplot(111)
    ax11.set_xlabel('y1', fontsize=15)
    ax11.set_ylabel('y2', fontsize=15)
    ax11.scatter(y1_values, y2_values)
    plt.show()


"""
#NSGA2的主函数
"""
def execute_nsga2_numpy(demands_np,DEMANDS_COUNT,access_type="g"):
    # 初始化种群 population = [[],[],[]]
    population_ar = initial_population()
    iteration_no = 0
    #大的循环
    while (iteration_no < ITERATION_NUM):
        print(iteration_no)
        # 生成两倍的后代，然后用于选择
        population_double=np.empty(shape=[POP_SIZE*2, SOLUTION_LEN],dtype=np.float32)
        population_double[0:POP_SIZE:1,:] = population_ar.copy()

        for i_new in range(POP_SIZE,POP_SIZE*2):
            a1 = random.randint(0, POP_SIZE - 1)
            b1 = random.randint(0, POP_SIZE - 1)
            # 通过crossover/limitation/mutation的方式生成新的soultion
            population_double[i_new,:]=crossover_mutation_limitation(population_ar[a1], population_ar[b1])
        # 评价适应度值
        y1_values_double, y2_values_double = fitness_numpy(demands_np, population_double,DEMANDS_COUNT,access_type=access_type)
        #test
        # 打印过程结果
        for index_x in range(0, POP_SIZE*2,4):
            print("x={0},y1={1},y2={2}".format(population_double[index_x], round(y1_values_double[index_x], 3),
                                               round(y2_values_double[index_x], 3)), end="\n")
        # 非支配解排序
        # 目标：y1--大，y2--小
        non_do_sorted_double_fronts = fast_non_dominated_sort(y1_values_double.copy(), y2_values_double.copy())
        # 拥挤度计算
        c_distance_double = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            c_distance_double.append(crowding_distance(y1_values_double.copy(), y2_values_double.copy(),non_do_sorted_double_fronts[i][:]))
        # 生成新的一代
        index_list_new_popu = []
        for i in range(0, len(non_do_sorted_double_fronts)):
            non_dominated_sorted_solution2_1 = [get_index_of(non_do_sorted_double_fronts[i][j], non_do_sorted_double_fronts[i]) for j in range(0, len(non_do_sorted_double_fronts[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], c_distance_double[i][:])
            front = [non_do_sorted_double_fronts[i][front22[j]] for j in range(0, len(non_do_sorted_double_fronts[i]))]
            front.reverse()
            for index in front:
                index_list_new_popu.append(index)
                if (len(index_list_new_popu) == POP_SIZE):
                    break
            if (len(index_list_new_popu) == POP_SIZE):
                break
        population_ar = [population_double[i] for i in index_list_new_popu]
        iteration_no = iteration_no + 1
    #结束大的循环

    # 输出当前代的结果，当前代是上代的最优结果
    y1_values, y2_values = fitness_numpy(demands_np,np.array(population_ar),DEMANDS_COUNT,access_type=access_type)
    # 打印结果
    for index_x in range(0,POP_SIZE):
        print("x={0},y1={1},y2={2}".format(population_ar[index_x], round(y1_values[index_x], 3), round(y2_values[index_x], 3)), end="\n")
    # 将最后一代的fitness结果打印出来
    draw_2d_plot( y1_values, y2_values)

"""
主函数，基于numpy，包括了MongoDB对象操作、调取NSGA2优化算法
"""
def main_function_numpy_gravity():
    # 初始化mongo对象
    mongo_operater_obj = a_mongo_operater.MongoOperater(DB_NAME, COLLECTION_NAME)
    # 获取到所有的记录
    demands_np = mongo_operater_obj.find_records_format_in_numpy_gravity(0, DEMANDS_COUNT, PROVIDERS_COUNT)
    # 创建和计算vj，便于后面可达性计算复用，放入了demands_np array中；
    calculate_provider_vj_numpy_gravity(demands_np, DEMANDS_COUNT, PROVIDERS_COUNT)
    # 调取NSGA2函数
    execute_nsga2_numpy(demands_np,DEMANDS_COUNT,access_type="g")




#公用的全局变量
#交叉算子概率
CROSSOVER_PROB_THRESHOLD =0.45
#变异算子概率
MUTATION_PROB__THRESHOLD=0.1
#拥挤度距离防止分母为0
DELATE=2e-7
#拥挤度距离极大值
DISTANCE_INFINTE=444444444
#解空间的约束系数，2030年的儿科医生数量不超过700,左右限制区间为690-710
MAX_X=10
#最低值限制
MIN_X=1
#增加医生的数量
LIMINTATION=100
# 种群规模，一般为双数，便于交叉操作
POP_SIZE=100
#进化代数
ITERATION_NUM=20
#一个染色体含多个DNA，一个DNA含多个Gene，按这样的组织逻辑，解的空间从一维延伸到了二维
#染色体长度（特指含多少个DNA）
SOLUTION_LEN=40   #1个solution 对应的1个 chromesome
#需求点，即小区，的个数
DEMANDS_COUNT=184
#供给点，即充电桩，的个数，与SOLUTION_COUNT保持一致
PROVIDERS_COUNT=40
#MONGODB数据库的配置
DB_NAME="admin"
COLLECTION_NAME="moo_ps"

if __name__=="__main__":
    main_function_numpy_gravity()
