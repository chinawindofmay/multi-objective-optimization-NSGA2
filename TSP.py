import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random

matplotlib.rcParams['font.family'] = 'STSong'

# 种群数
POPULATION_SIZE = 300
# 改良次数
IMPROVE_COUNT = 100
# 进化次数
ITERATION = 300
# 设置强者的定义概率，即种群前30%为强者
RETAIN_RATE = 0.3
# 设置弱者的存活概率
SELECT_RATE = 0.5
# 变异率
MUTATION_RATE = 0.1

# 总距离
def get_total_distance(x,origin_id,points_distance):
    distance = 0
    distance += points_distance[origin_id][x[0]]
    for i in range(len(x)):
        if i == len(x) - 1:
            distance += points_distance[origin_id][x[i]]
        else:
            distance += points_distance[x[i]][x[i + 1]]
    return distance


# 改良
def improve(x,origin_id,points_distance):
    i = 0
    distance = get_total_distance(x,origin_id,points_distance)
    while i < IMPROVE_COUNT:
        # randint [a,b]
        u = random.randint(0, len(x) - 1)
        v = random.randint(0, len(x) - 1)
        if u != v:
            new_x = x.copy()
            t = new_x[u]
            new_x[u] = new_x[v]
            new_x[v] = t
            new_distance = get_total_distance(new_x,origin_id,points_distance)
            if new_distance < distance:
                distance = new_distance
                x = new_x.copy()
        else:
            continue
        i += 1


# 自然选择
def selection(population,origin_id):
    """
    选择
    先对适应度从大到小排序，选出存活的染色体
    再进行随机选择，选出适应度虽然小，但是幸存下来的个体
    """
    # 对总距离从小到大进行排序
    graded = [[get_total_distance(x,origin_id,points_distance), x] for x in population]
    graded = [x[1] for x in sorted(graded)]
    # 选出适应性强的染色体
    retain_length = int(len(graded) * RETAIN_RATE)
    parents = graded[:retain_length]
    # 选出适应性不强，但是幸存的染色体
    for chromosome in graded[retain_length:]:
        if random.random() < SELECT_RATE:
            parents.append(chromosome)
    return parents


# 交叉繁殖
def crossover(parents):
    # 生成子代的个数,以此保证种群稳定
    target_count = POPULATION_SIZE - len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male) - 1)

            # 交叉片段
            gene1 = male[left:right]
            gene2 = female[left:right]

            child1_c = male[right:] + male[:right]
            child2_c = female[right:] + female[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            for o in gene2:
                child1_c.remove(o)

            for o in gene1:
                child2_c.remove(o)

            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children


# 变异
def mutation(children):
    for i in range(len(children)):
        if random.random() < MUTATION_RATE:
            child = children[i]
            u = random.randint(1, len(child) - 4)
            v = random.randint(u + 1, len(child) - 3)
            w = random.randint(v + 1, len(child) - 2)
            child = children[i]
            child = child[0:u] + child[v:w] + child[u:v] + child[w:]


# 得到最佳纯输出结果
def get_fitness(population, origin_id):
    graded = [[get_total_distance(x,origin_id,points_distance), x] for x in population]
    graded = sorted(graded)
    return graded[0][0], graded[0][1]


def genetic(points_location, points_distance,origin_id):
    # 设置起点
    # origin_id = 15
    points_count=len(points_location)
    index = [i for i in range(points_count)]
    index.remove(origin_id)

    # 使用改良圈算法初始化种群
    population = []
    for i in range(POPULATION_SIZE):
        # 随机生成个体
        x = index.copy()
        random.shuffle(x)
        improve(x,origin_id,points_distance)
        population.append(x)

    register = []
    i = 0
    distance, result_path = get_fitness(population, origin_id)
    while i < ITERATION:
        # 选择繁殖个体群
        parents = selection(population,origin_id)
        # 交叉繁殖
        children = crossover(parents)
        # 变异操作
        mutation(children)
        # 更新种群
        population = parents + children
        # 评估值
        distance, result_path = get_fitness(population, origin_id)
        register.append(distance)
        i = i + 1

    print(distance)
    print(result_path)

    result_path = [origin_id] + result_path + [origin_id]
    X = []
    Y = []
    for index in result_path:
        X.append(points_location[index, 0])
        Y.append(points_location[index, 1])

    #将结果展示出来
    plt.plot(X, Y, '-o')
    plt.plot(points_location[origin_id,0],points_location[origin_id,1],"om")
    plt.show()

    #将收敛过程展示出来
    plt.plot(list(range(len(register))), register)
    plt.show()



def read_data(file_path='./tsp_data.txt'):
    # 载入数据
    points_location = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            line = line.split(',')
            points_location.append([float(line[1]), float(line[2])])
    points_location = np.array(points_location)

    # 展示地图
    # plt.scatter(city_condition[:,0],city_condition[:,1])
    # plt.show()

    # 距离矩阵
    points_count = len(points_location)
    points_distance = np.zeros([points_count, points_count])
    for i in range(points_count):
        for j in range(points_count):
            points_distance[i][j] = math.sqrt(
                (points_location[i][0] - points_location[j][0]) ** 2 + (points_location[i][1] - points_location[j][1]) ** 2)
    return  points_location, points_distance


#入口
if __name__=="__main__":
    points_location, points_distance = read_data()
    genetic(points_location, points_distance,origin_id=15)
