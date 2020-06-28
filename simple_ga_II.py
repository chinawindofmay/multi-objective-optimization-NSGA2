#coding:utf-8

import numpy
import matplotlib.pyplot as plt

'''
参考链接：https://www.cnblogs.com/xxhbdk/p/9368388.html

遗传算法特征：自由空间，定长编码
选择：择优选择
交叉：全空间可遍历
变异：增强全空间的搜索能力

亮点：采用十进制编码，没不需要进行编码解码工作

疑惑点：选择：这里的轮盘间隔的计算方式是什么逻辑
              解答:说好的是择优选择，为什么适应度要做降序排列，岂不是值越优，取到的概率越小（这里的问题是计算目标函数的最小值）
        种群的收敛速度
        搜索空间的维度
              
'''

# 目标函数1
def myfunc1(x):
   return (x ** 2 - 5 * x) * numpy.sin(x ** 2) * -1

# 目标函数2
# def myfunc2(x1, x2):
    # return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


# 遗传算法: 选择, 交叉, 变异
class GA(object):

    def __init__(self, func, lbounds, ubounds, population_size, maxIter, pm, speed, cf):
        self.func = func                           # 目标函数
        self.lbounds = lbounds                     # 搜寻下界
        self.ubounds = ubounds                     # 搜寻上界
        self.population_size = population_size     # 种群规模
        self.maxIter = maxIter                     # 最大迭代次数
        self.pm = pm                               # 变异概率(0, 1)
        self.speed= speed                           # 种群收敛速度[1, +∞)
        self.cf = cf                               # 交叉因子(0, 1)

        self.size = len(lbounds)                   # 搜索空间的维度
        self.best_chrom_fitness = list()           # 最优染色体(染色体, 适应度)迭代记录列表
        self.__record_fitness = list()             # 种群适应度迭代记录列表

    def solve(self):
        # 种群随机初始化
        population = self.__init_population()
        # 记录种群最优染色体信息
        self.__add_best(population)

        for i in range(self.maxIter):
            # 种群更新
            population = self.__selection(population)
            population = self.__crossover(population)
            population = self.__mutation(population)

            # 上一代最优个体无条件保留至下一代。每循环一此在best_chrom_fitness末尾追加一个最优个体
            population[0] = self.best_chrom_fitness[-1][0] #self.best_chrom_fitness.append((population[min_idx], fitness[min_idx]))
            # 记录种群最优个体
            self.__add_best(population)
        self.solution = self.best_chrom_fitness[-1]#最优解

    # 选择: 轮盘赌方法
    def __selection(self, population):
        # 适应度排序
        fitness = self.__cal_fitness(population)#列表
        #sorted 按照适应度排序
        # reverse = True 降序    按照每个item的items[0]
        # list((ele, idx) for idx, ele in enumerate(fitness))列表中嵌套元组，每个元组格式（适应度，索引）
        new_fitness = sorted(list((ele, idx) for idx, ele in enumerate(fitness)), key=lambda item: item[0], reverse=True)
        # 轮盘区间计算 -> 采用多项式函数对收敛速度进行调整
        roulette_interval = self.__cal_interval()
        # 随机飞镖排序
        random_dart = sorted(numpy.random.random(self.population_size))#升序排列,0.0~1.0之间的100个随机数
        new_population = list()
        idx_interval = idx_dart = 0
        while idx_dart < self.population_size:
            if random_dart[idx_dart] > roulette_interval[idx_interval]:
                idx_interval += 1
            else:
                new_population.append(population[new_fitness[idx_interval][1]])
                idx_dart += 1

        # 顺序打乱
        numpy.random.shuffle(new_population)
        return new_population

    # 交叉: 对称凸组合
    def __crossover(self, population):
        # 交叉随机数 -> 采用交叉因子提高计算精度
        alpha = numpy.random.random(self.population_size - 1) * self.cf#产生99个随机数，每个随机数范围在[0,0.1）

        for idx in range(self.population_size - 1):
            new_chrom1 = alpha[idx] * population[idx] + (1 - alpha[idx]) * population[idx + 1]
            new_chrom2 = alpha[idx] * population[idx + 1] + (1 - alpha[idx]) * population[idx]
            population[idx] = new_chrom1
            population[idx + 1] = new_chrom2

        return population

    # 变异: 全空间变异
    def __mutation(self, population):
        # 变异概率随机数
        mutation_prob = numpy.random.random(self.population_size)
        for idx, prob in enumerate(mutation_prob):
            if prob <= self.pm:
                # 变异幅度随机数
                mutation_amplitude = numpy.random.uniform(-1, 1, self.size)#[-1,1）范围内均匀分布的随机数
                for idx_dim, ampli in enumerate(mutation_amplitude):
                    if ampli >= 0:    # 正向变异
                        population[idx][idx_dim] += ampli * (self.ubounds[idx_dim] - population[idx][idx_dim])
                    else:             # 负向变异
                        population[idx][idx_dim] += ampli * (population[idx][idx_dim] - self.lbounds[idx_dim])

        return population

    # 种群随机初始化
    def __init_population(self):
        population = list()
        for i in range(self.population_size):
            chrom = list()
            for j in range(self.size):
                #从一个均匀分布[low,high)中随机采样
                chrom.append(numpy.random.uniform(self.lbounds[j], self.ubounds[j]))
            population.append(numpy.array(chrom))
            #population.append(chrom)
        #return numpy.array(population)
        return population

    # 种群适应度计算
    def __cal_fitness(self, population):
        #不确定有多少个参数就在变量前面加上*，无论有多少个，在函数内部都被存放在以形参名为标识符的tuple中。
        fitness = list(self.func(*chrom) for chrom in population)
        return fitness

    # 记录种群最优染色体信息
    def __add_best(self, population):
        fitness = self.__cal_fitness(population)
        self.__record_fitness.append(fitness)
        min_idx = numpy.argmin(fitness)#返回适应度最小值的索引
        self.best_chrom_fitness.append((population[min_idx], fitness[min_idx]))

    # 轮盘区间计算
    def __cal_interval(self):
        #保证轮盘区间边界是0~1之间
        tmp = (numpy.arange(self.population_size) + 1) / self.population_size # (0,1]
        tmp_normalize = tmp / (self.population_size + 1) * 2#每个元素选中的概率之间是等差数列
        roulette_interval = list()
        curr_sum = 0
        for item in tmp_normalize:
            curr_sum += item
            roulette_interval.append(curr_sum)
        roulette_interval = numpy.array(roulette_interval) ** self.speed
        return roulette_interval

    # 求解过程可视化展示
    def display(self):
        fig = plt.figure(figsize=(8, 5))
        axes = plt.subplot()
        # arra=numpy.array(self.__record_fitness)
        # print(arra.shape)#(1001, 100)因为在开始循环的时候就添加了一行
        axes.plot(self.__record_fitness, 'g.')#每次循环的目标函数值

        #目标函数值的平均值
        axes.plot(numpy.array(self.__record_fitness).sum(axis=1) / self.population_size, 'r-', label='$meanVal$')# 'r-'表示红色的实线
        #每次循环中的最优解
        axes.plot(numpy.array(self.best_chrom_fitness)[:,[1]], 'b--', label="bestFitness")#波动不是很大
        axes.set(xlim=(-1, self.maxIter+1), xlabel='$iterCount$', ylabel='$fitness$')
        axes.set(title = 'solution = {}'.format(self.solution))
        axes.legend()
      #  fig.savefig('myGA.png', dpi=500)
        plt.show()
        plt.close()


if __name__ == '__main__':
    ins = GA(myfunc1, [-9], [5], population_size=100, maxIter=1000, pm=0.01, speed=1, cf=0.1)
    # ins = GA(myfunc2, [-10, -10], [10, 10], population_size=100, maxIter=500, pm=0.3, speed=1, cf=0.1)
    ins.solve()
    ins.display()