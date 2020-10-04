import numpy as np


def initial_dsf_without_choice(low,up,x_dim,E_setting):
    solution = np.random.randint(low=low, high=up + 1,size=(x_dim))  # 新建的设施
    print("第1步：初始随机个体的基因值")
    print(solution)
    print(np.sum(solution))
    solution = (solution * (E_setting / np.sum(solution))).astype(np.int)  # 规模介于up和low之间
    print("第2步：最大值整数约束")
    print(solution)
    print(np.sum(solution))
    # 会存在部分因最大值整数约束，而超限的数值，需要先处理到【low,up】范围内
    while np.sum(solution[solution<low])>0 or np.sum(solution[solution>up])>0:
        solution[solution<low]+=1
        solution[solution>up]-=1
    print("第3步：超限的数值处理到【low,up】范围内")
    print(solution)
    print(np.sum(solution))
    # 补齐因最大值整数约束导致的部分值丢失
    #该方法对于极限情况，如：30,100，会存在多次循环的问题，所以增加一个判别条件，让其不至于循环很久，提高效率，但是这种方法有极小的可能性会导致值不准确，目前忽略未用
    iteation_count=0
    while np.sum(solution) != E_setting:
        delate = np.sum(solution) - E_setting
        if delate > 0:
            adjust_mark = -1
        else:
            adjust_mark = +1
        for adjust_i in range(np.abs(delate)):
            adjust_index = np.random.choice(a=[i for i in range(x_dim)], size=1, replace=False)
            if solution[adjust_index] > low and solution[adjust_index] < up:
                solution[adjust_index] += adjust_mark
            if delate > 0 and solution[adjust_index] == up:
                solution[adjust_index] += adjust_mark
            if delate < 0 and solution[adjust_index] == low:
                solution[adjust_index] += adjust_mark
        # if iteation_count<100:
        #     iteation_count+=1
        # else:
        #     break
    print("第4步：补齐因最大值整数约束导致的部分值丢失")
    print(solution)
    print(np.sum(solution))
    return solution

def test_initial_dsf_without_choice():
    #总和值大于等于30，小于等于100
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,30)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,33)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,36)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,50)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,60)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,70)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,94)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,99)
    print("#####################################################")
    initial_dsf_without_choice(3,10,10,100)


def initial_dsf_with_choice(low,up,x_dim,E_setting,k):
    #选择出k个盘子
    choic_index_list = np.random.choice(a=[i for i in range(x_dim)], size=k,
                                        replace=False)
    creating_mark = np.full(shape=(x_dim), fill_value=0)
    creating_mark[choic_index_list] = 1
    solution = np.random.randint(low=low, high=up + 1,size=(x_dim))  # 新建的设施
    solution=creating_mark*solution
    print("第1步：初始随机个体的基因值")
    print(solution)
    print(np.sum(solution))
    solution = (solution * (E_setting / np.sum(solution))).astype(np.int)  # 规模介于up和low之间
    print("第2步：最大值整数约束")
    print(solution)
    print(np.sum(solution))
    # 会存在部分因最大值整数约束，而超限的数值，需要先处理到【low,up】范围内
    if np.sum(solution[np.where((solution<low) & (solution!=0))])>0:
        print("test")
    while np.sum(solution[np.where((solution<low) & (solution!=0))])>0 or np.sum(solution[solution>up])>0:
        solution[np.where((solution<low) & (solution!=0))]+=1
        solution[solution>up]-=1
    print("第3步：超限的数值处理到【low,up】范围内")
    print(solution)
    print(np.sum(solution))
    # 补齐因最大值整数约束导致的部分值丢失
    #该方法对于极限情况，如：30,100，会存在多次循环的问题，所以增加一个判别条件，让其不至于循环很久，提高效率，但是这种方法有极小的可能性会导致值不准确，目前忽略未用
    iteation_count=0
    while np.sum(solution) != E_setting:
        delate = np.sum(solution) - E_setting
        if delate > 0:
            adjust_mark = -1
        else:
            adjust_mark = +1
        for adjust_i in range(np.abs(delate)):
            adjust_index = np.random.choice(a=choic_index_list, size=1, replace=False)
            if solution[adjust_index] > low and solution[adjust_index] < up:
                solution[adjust_index] += adjust_mark
            if delate > 0 and solution[adjust_index] == up:
                solution[adjust_index] += adjust_mark
            if delate < 0 and solution[adjust_index] == low:
                solution[adjust_index] += adjust_mark
        # if iteation_count<100:
        #     iteation_count+=1
        # else:
        #     break
    print("第4步：补齐因最大值整数约束导致的部分值丢失")
    print(solution)
    print(np.sum(solution))
    return solution

def test_initial_dsf_with_choice():
    #总和值大于等于30，小于等于100
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,30,8)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,33,4)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,36,5)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,50,8)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,60,6)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,70,8)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,94,10)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,99,10)
    print("#####################################################")
    initial_dsf_with_choice(3,10,10,100,10)


def calculate_symmetry_indicator( frequency_result):
    ## 首先定义点序列
    symmetry_keyid_array = np.array([[1, 3, 4, 8, 10, 15], [2, 7, 6, 9, 14, 16]])
    ## 将点对的评率填充到上面来
    frequency_cf_vector = np.full(shape=symmetry_keyid_array.shape, fill_value=0)
    for ki in range(frequency_cf_vector.shape[0]):
        for kj in range(frequency_cf_vector.shape[1]):
            temp = 0
            for i in range(frequency_result.shape[0]):
                for j in range(frequency_result.shape[1] - 1):
                    if frequency_result[i, j] == symmetry_keyid_array[ki, kj]:
                        temp += frequency_result[i, -1]
            frequency_cf_vector[ki, kj] = temp
    print(frequency_cf_vector)
    ## 其次计算相关系数值
    print("迭代次数{0}，种群规模{1}，其对称相关系数值为：{2}".format(20, 500, np.corrcoef(frequency_cf_vector)))

def teest_calculate_symmetry_indicator():
    frequency_result = np.loadtxt('./frequency_result.txt', delimiter=',')
    calculate_symmetry_indicator(frequency_result)
if __name__=="__main__":
    # test_initial_dsf_with_choice()
    teest_calculate_symmetry_indicator()