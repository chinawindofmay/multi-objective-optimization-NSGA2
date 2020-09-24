import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt


class SPEA2():
    def __init__(self, dim, pop, max_iter):  # 维度，群体数量，迭代次数
        self.pc = 0.4  # 交叉概率
        self.pm = 0.4  # 变异概率
        self.dim = dim  # 搜索维度
        self.pop = pop  # 粒子数量
        self.K = int(np.sqrt(pop + pop))  # 距离排序，第k个距离值
        self.max_iter = max_iter  # 迭代次数
        self.population = []  # 父代种群
        self.archive = []  # 存档集合
        self.popu_arch = []  # 合并后的父代与存档集合种群
        # self.fronts = []                        #Pareto前沿面
        self.KNN = []  # 最近领域距离，K-th
        # self.rank = []#np.zeros(self.pop)       #非支配排序等级
        self.S = []  # 个体 i的 Strength Value
        self.D = []  # density，距离度量
        self.R = []  # 支配关系度量
        self.F = []  # 适应度
        self.objectives = []  # 目标函数值
        # self.np = []                            #该个体支配的其它个体数目
        self.set = []  # 被支配的个体集

    def init_Population(self):  # 初始化种群
        self.population = np.zeros((self.pop, self.dim))
        self.archive = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for j in range(self.dim):
                self.population[i][j] = random.random()
                self.archive[i][j] = random.random()

    def popu_archive(self):  # Population和 Archive合并,pop*2
        self.popu_arch = np.zeros((2 * self.pop, self.dim))
        for i in range(self.pop):
            for j in range(self.dim):
                self.popu_arch[i][j] = self.population[i][j]
                self.popu_arch[i + self.pop][j] = self.archive[i][j]

    def cal_obj(self, position):  # 计算一个个体的多目标函数值 f1,f2 最小值
        f1 = position[0]
        f = 0
        for i in range(self.dim - 1):
            f += 9 * (position[i + 1] / (self.dim - 1))
        g = 1 + f
        f2 = g * (1 - np.square(f1 / g))
        return [f1, f2]

    def cal_fitness(self):  # 计算 Pt和 Et 适应度, F(i) = R(i) + D(i)
        self.objectives = []
        self.set = []
        self.S = np.zeros(2 * self.pop)
        self.D = np.zeros(2 * self.pop)
        self.R = np.zeros(2 * self.pop)
        self.F = np.zeros(2 * self.pop)
        self.KNN = np.zeros(2 * self.pop)
        position = []
        for i in range(2 * self.pop):
            position = self.popu_arch[i]
            self.objectives.append(self.cal_obj(position))
        # 计算 S 值
        for i in range(2 * self.pop):
            temp = []
            for j in range(2 * self.pop):
                if j != i:
                    if self.objectives[i][0] <= self.objectives[j][0] and self.objectives[i][1] <= self.objectives[j][
                        1]:
                        self.S[i] += 1  # i支配 j，np+1
                    if self.objectives[j][0] <= self.objectives[i][0] and self.objectives[j][1] <= self.objectives[i][
                        1]:
                        temp.append(j)  # j支配 i
            self.set.append(temp)
            # 计算 R 值
        for i in range(2 * self.pop):
            for j in range(len(self.set[i])):
                self.R[i] += self.S[self.set[i][j]]
        # 计算 D 值
        for i in range(2 * self.pop):
            distance = []
            for j in range(2 * self.pop):
                if j != i:
                    distance.append(np.sqrt(np.square(self.objectives[i][0] - self.objectives[j][0]) + np.square(
                        self.objectives[i][1] - self.objectives[j][1])))
            distance = sorted(distance)
            self.KNN[i] = distance[self.K - 1]  # 其它个体与个体 i 的距离，升序排序，取第 K 个距离值
            self.D[i] = 1 / (self.KNN[i] + 2)
        # 计算 F 值
        for i in range(2 * self.pop):
            self.F[i] = self.D[i] + self.R[i]

    def update(self):  # 下一代 Archive
        # self.archive = []
        juli = []
        shiyingzhi = []
        a = 0
        for i in range(2 * self.pop):
            if self.F[i] < 1:
                juli.append([self.D[i], i])
                a = a + 1
            else:
                shiyingzhi.append([self.F[i], i])
        # 判断是否超出范围
        if a > self.pop:  # 截断策略
            juli = sorted(juli)
            for i in range(self.pop):
                self.archive[i] = self.popu_arch[juli[i][1]]
        if a == self.pop:  # 刚好填充
            for i in range(self.pop):
                self.archive[i] = self.popu_arch[juli[i][1]]
        if a < self.pop:  # 适应值筛选
            shiyingzhi = sorted(shiyingzhi)
            for i in range(a):
                self.archive[i] = self.popu_arch[juli[i][1]]
            for i in range(self.pop - a):
                self.archive[i + a] = self.popu_arch[shiyingzhi[i][1]]

    def cal_fitness2(self):  # 计算 Pt和 Et 适应度, F(i) = R(i) + D(i)
        self.objectives = []
        self.set = []
        self.S = np.zeros(self.pop)
        self.D = np.zeros(self.pop)
        self.R = np.zeros(self.pop)
        self.F = np.zeros(self.pop)
        self.KNN = np.zeros(self.pop)
        position = []
        for i in range(self.pop):
            position = self.archive[i]
            self.objectives.append(self.cal_obj(position))
        # 计算 S 值
        for i in range(self.pop):
            temp = []
            for j in range(self.pop):
                if j != i:
                    if self.objectives[i][0] <= self.objectives[j][0] and self.objectives[i][1] <= self.objectives[j][
                        1]:
                        self.S[i] += 1  # i支配 j，np+1
                    if self.objectives[j][0] <= self.objectives[i][0] and self.objectives[j][1] <= self.objectives[i][
                        1]:
                        temp.append(j)  # j支配 i
            self.set.append(temp)
            # 计算 R 值
        for i in range(self.pop):
            for j in range(len(self.set[i])):
                self.R[i] += self.S[self.set[i][j]]
        # 计算 D 值
        for i in range(self.pop):
            distance = []
            for j in range(self.pop):
                if j != i:
                    distance.append(np.sqrt(np.square(self.objectives[i][0] - self.objectives[j][0]) + np.square(
                        self.objectives[i][1] - self.objectives[j][1])))
            distance = sorted(distance)
            self.KNN[i] = distance[self.K - 1]  # 其它个体与个体 i 的距离，升序排序，取第 K 个距离值
            self.D[i] = 1 / (self.KNN[i] + 2)
        # 计算 F 值
        for i in range(self.pop):
            self.F[i] = self.D[i] + self.R[i]  # 适应度越小越好

    def selection(self):  # 轮盘赌选择
        pi = np.zeros(self.pop)  # 个体的概率
        qi = np.zeros(self.pop + 1)  # 个体的累积概率
        P = 0
        self.cal_fitness2()  # 计算Archive的适应值
        for i in range(self.pop):
            P += 1 / self.F[i]  # 取倒数，求累积适应度
        for i in range(self.pop):
            pi[i] = (1 / self.F[i]) / P  # 个体遗传到下一代的概率
        for i in range(self.pop):
            qi[0] = 0
            qi[i + 1] = np.sum(pi[0:i + 1])  # 累积概率
        # 生成新的 population
        self.population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            r = random.random()  # 生成随机数，
            for j in range(self.pop):
                if r > qi[j] and r < qi[j + 1]:
                    self.population[i] = self.archive[j]
                j += 1

    def crossover(self):  # 交叉,SBX交叉
        for i in range(self.pop - 1):
            # temp1 = []
            # temp2 = []
            if random.random() < self.pc:
                # pc_point = random.randint(0,self.dim-1)        #生成交叉点
                # temp1.append(self.population[i][pc_point:self.dim])
                # temp2.append(self.population[i+1][pc_point:self.dim])
                # self.population[i][pc_point:self.dim] = temp2
                # self.population[i+1][pc_point:self.dim] = temp1
                a = random.random()
                for j in range(self.dim):
                    self.population[i][j] = a * self.population[i][j] + (1 - a) * self.population[i + 1][j]
                    self.population[i + 1][j] = a * self.population[i + 1][j] + (1 - a) * self.population[i][j]
            i += 2

    def mutation(self):  # 变异
        for i in range(self.pop):
            for j in range(self.dim):
                if random.random() < self.pm:
                    self.population[i][j] = self.population[i][j] - 0.1 + np.random.random() * 0.2
                    if self.population[i][j] < 0:
                        self.population[i][j] = 0  # 最小值0
                    if self.population[i][j] > 1:
                        self.population[i][j] = 1  # 最大值1

    def draw(self):  # 画图
        self.cal_fitness2()
        self.objectives = []
        a = 0
        for i in range(self.pop):
            if self.F[i] < 1:  # 非支配个体
                a += 1
                position = self.archive[i]
                self.objectives.append(self.cal_obj(position))
        x = []
        y = []
        for i in range(a):
            x.append(self.objectives[i][0])
            y.append(self.objectives[i][1])
        ax = plt.subplot(111)
        plt.scatter(x, y)  # ,marker='+')#self.objectives[:][0],self.objectives[:][1]) #?
        # plt.plot(,'--',label='')
        plt.axis([0.0, 1.0, 0.0, 1.1])
        xmajorLocator = MultipleLocator(0.1)
        ymajorLocator = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('ZDT2 Pareto Front')
        plt.grid()
        # plt.show()
        plt.savefig('SPEA ZDT2 Pareto Front 2.png')

    def run(self):  # 主程序
        self.init_Population()  # 初始化种群，选择交叉变异，生成子代种群
        self.popu_archive()
        self.cal_fitness()
        self.update()
        for i in range(self.max_iter - 1):
            self.selection()
            self.crossover()
            self.mutation()
            self.popu_archive()
            self.cal_fitness()
            self.update()
            print(i)
            # self.selection()
            # self.crossover()
            # self.mutation()
        self.draw()


def main():
    SPEA = SPEA2(30, 100, 500)
    SPEA.run()


if __name__ == '__main__':
    main()