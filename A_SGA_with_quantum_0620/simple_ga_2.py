import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

'''
参考链接：https://blog.csdn.net/ha_ha_ha233/article/details/91364937

存在缺陷：
交叉，没有考虑父染色体和母染色体是同一染色体的情况；
变异，仅仅是X上发生变异，没有考虑Y上会发生变异；
单点交叉，单点变异；

有一点疑问，在交叉变异运算中，一直这样直接等来等去，在改变的时候会不会影响源数据（好像也没太大关系，对后面计算没什么影响）

'''

DNA_SIZE = 24#染色体长度
POP_SIZE = 200#种群规模
CROSSOVER_RATE = 0.8#交叉概率
MUTATION_RATE = 0.01#变异概率
N_GENERATIONS = 500#迭代次数
X_BOUND = [-5, 5]
Y_BOUND = [-5, 5]


def F222(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)
    # return -(20+x**2+y**2-10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))

#绘制3D图形
def plot_3d(ax,F):
    X = np.linspace(*X_BOUND, 100)#等差数列
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)#坐标矩阵，每个交叉点对应网格
    Z = F(X, Y)
    # 绘制3D图形，X,Y,Z:2D数组形式的数据值
    #rstride:数组行距（步长大小）
    #cstride:数组列距(步长大小）
    #cmap:曲面块颜色映射
    #color:曲面块颜色
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-100, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)#不仅可以绘图而且鼠标可以继续交互
    plt.show()


### 2.5 画出适应度函数值变化图
def plot(results):
    '''画图
    '''
    X = []
    Y = []
    for i in range(N_GENERATIONS):
        X.append(i + 1)
        Y.append(results[i])
    plt.plot(X, Y)
    plt.xlabel('Number of iteration', size=15)
    plt.ylabel('Value', size=15)
    plt.title('GA')
    plt.show()

#计算适应度函数
def get_fitness(pop,F):
    x, y = translateDNA(pop)
    z = F(x, y)
    #为了防止0的出现（后面计算概率会有问题），可以在后面加上一个很小很小接近于0的值
    return z,(z - np.min(z)+1e-3)  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]

#解码
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目,奇数列
    #表示x，偶数列表示y，（200,48）
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, 0::2]  # 偶数列表示y
    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    #...[::-1]可以理解成一个倒序
    #2 ** np.arange(DNA_SIZE)[::-1]，2的n次幂
    #x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1])对应个体的十进制编码

    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]

    return x, y

#交叉和变异
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        # np.random.rand()通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 在种群中选择另一个个体，并将该个体作为母亲
            #为什么随机数的最大值限制要 * 2，交叉点可能在X也可能是在Y
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因，替换交叉点之后的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

#变异
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        #这里没有*2？？？
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        #异或，相同为0，相异为1
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

#选择操作:通过随机操作，保留个体，规模仍与原pop相同
def select(pop, fitness):  # nature selection wrt pop's fitness
    #产生随机数，replace=True可以取相同数字，p实际上是一个数组，大小（size）与指定的pop_size相同，用来规定选取数组中每个元素的概率，默认为概率相同。
    #返回的值为shape为（200，）的索引数组
    #大致意思就是从np.arange(POP_SIZE)中随机抓取一个元素，概率为p,将pop[idx]作为保留的个体
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]

def print_info(pop,F):
    z,fitness = get_fitness(pop,F)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


def traditional_ga(F):
    z_results = []
    best_z = -1000000
    Z_result = []
    Z_mean_result = []
    Z_median_result = []
    Z_max_result = []
    # 编码
    # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目，DNA_SIZE为编码长度。
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE*2)#初始化群体的生产
    for iii in range(N_GENERATIONS):  # 迭代N代
        print(iii)
        x, y = translateDNA(pop)  # 二进制解码
        # locals() 函数会以字典类型返回当前位置的全部局部变量。
        # if 'sca' in locals():
        #     sca.remove()
        # sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        # plt.show();#显示所绘制的图像
        # plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))  # 交叉和变异
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        Z, fitness = get_fitness(pop, F)  # 计算适应度
        Z_result.append(Z)
        #  Z_mean= np.sum(Z)/POP_SIZE
        Z_mean = np.mean(Z)
        Z_median=np.median(Z)
        Z_max = np.max(Z)
        Z_mean_result.append(Z_mean)
        Z_median_result.append(Z_median)
        Z_max_result.append(Z_max)
        z = np.max(Z)
        ## 找出到目前为止最优的适应度函数值和对应的参数
        if z > best_z:
            best_z = z
        z_results.append(best_z)
        pop = select(pop, fitness)  # 选择生成新的种群
    print_info(pop, F)
    return Z_max_result,Z_mean_result,Z_median_result,Z_result
if __name__ == "__main__":
    #绘制函数三维图像
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax,F222)
    Z_max_result,Z_mean_result,Z_median_result,Z_result=traditional_ga(F222)

    #如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。
    plt.ioff()
    plot_3d(ax,F222)
    plot(Z_result)
    #二维可视化结果
    axes = plt.subplot()
    axes.plot(Z_result, 'g.')  # 每次循环的目标函数值
    # axes.plot( Z_mean_result,'r-',label="mean_funcValue")
    # axes.plot(Z_max_result, 'b-', label="max_funcValue")
   # axes.plot(z_results, 'b--', label="best_fitness")
    axes.set(xlim=(-1,N_GENERATIONS + 1), xlabel='$iterCount$', ylabel='$value$')
    axes.legend()
    plt.show()
    # plot(Z/POP_SIZE)