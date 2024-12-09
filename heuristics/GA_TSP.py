'''以下是使用遗传算法求解TSP问题的 Python 代码示例：
该算法的思路是通过遗传算法不断地进行“繁殖”，从而得到更优秀的解。
它的基本流程分为初始化种群、适应度评估、选择、交叉和变异操作、更新种群和输出最优解等步骤。
在该代码中，输入数据包括城市数量 `city_number = 10 ` 和城市位置坐标 `city_position = []`，输出数据包括最短路径和最优解。
'''
import copy
from tqdm import tqdm
import time
'''
一、设置城市数量city_number,然后根据随机数生成城市的位置city_position,然后计算城市之间的距离矩阵dis_mat
二、初始化种群，根据城市数量和种群个数来生成种群个数，此时种群个体表示的是路径
三、根据种群路径和城市之间的距离计算每条路径的适应值---方法：直根据路径进行城市距离的求和，然后取平方的倒数作为适应值、
四、进行选择操作---
    1. 首先，生成一个随机整数 index，这个整数的取值范围是从0到种群大小减1。
    2. 接着，初始化一个累加量 s，然后从头到尾遍历种群中的所有个体，依次将每个个体的适应度分值加到 s 中。
    3. 在每次循环中，判断当前 s 是否大于等于一个随机数 r，如果是，则返回该个体，否则继续循环。
    4. 返回选择出来的个体。
五、进行变异操作---
    1. 首先，判断是否进行变异操作。如果随机生成的一个小数值小于变异概率 pm，则进行变异操作，否则不进行任何操作。
    2. 然后，生成两个变异位置 mpoint1 和 mpoint2，这两个位置都是随机整数，并且取值范围是0到城市数减1。
    3. 最后，将变异位置上的两个基因位进行交换，从而生成一个新的个体。
六、进行迭代计算
'''
import random
import math
import numpy as np

from argparse import ArgumentParser


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--N", type=int, default=10)

    args = parser.parse_args()
    return args

# 初始化种群并去重
def initpopulation(pop_size, city_number):
    popuation = []
    while len(popuation) < pop_size:
        temp = random.sample(range(city_number), city_number)
        if temp not in popuation:
            popuation.append(temp)
    return popuation


def fitness(population, dis_mat):
    fitness = []
    for i in range(len(population)):
        distance = 0.0
        for j in range(city_number - 1):
            distance += dis_mat[population[i][j]][population[i][j + 1]]
        distance += dis_mat[population[i][-1]][population[i][0]]
        if distance == 0:
            f = float('inf')  # 如果路径长度为 0，则设置适应度为一个较大的数
        else:
            f = 1 / distance ** 2  # 计算路径长度平方的倒数作为适应度
        fitness.append(f)

    return fitness


# 选择函数，轮盘赌选择
def select(population, fitness):
    index = random.randint(0, pop_size - 1)
    s = 0
    r = random.uniform(0, sum(fitness))
    for i in range(len(population)):
        s += fitness[i]
        if s >= r:
            index = i
            break
    return population[index]


def crossover(parent1, parent2):  # 传入两个父代
    if random.random() < pc:  # 按照一定的概率进行交叉操作
        chrom1 = parent1[:]  # 复制父代染色体
        chrom2 = parent2[:]
        # 交叉点，选择两个随机的交叉点。如果第一个点在第二个点的右侧，则交换两个点
        cpoint1 = random.randint(0, city_number - 1)
        cpoint2 = random.randint(0, city_number - 1)
        if cpoint1 > cpoint2:
            temp = cpoint1
            cpoint1 = cpoint2
            cpoint2 = temp

        # 未进行杂交之前，先保存两个父代个体的杂交段以及杂交段后面的片段
        # 保存cpoint1以及后面的片段
        temp1 = []
        temp2 = []
        for i in range(cpoint1, len(chrom1)):
            temp1.append(chrom1[i])
            temp2.append(chrom2[i])

        # 交叉操作，在交叉点之间对染色体进行交叉操作。
        for i in range(cpoint1, cpoint2 + 1):
            chrom1[i] = parent2[i]
            chrom2[i] = parent1[i]

        # 在杂交之后，先只保留每个父体杂交段以及杂交段以前的片段，然后在加上未杂交之前准备的杂交段以及杂交段后面的片段
        # 保存cpoint2以及前面的片段
        new_chrom1 = []
        new_chrom2 = []
        for i in range(cpoint2 + 1):
            new_chrom1.append(chrom1[i])
            new_chrom2.append(chrom2[i])
        new_chrom1.extend(temp1)
        new_chrom2.extend(temp2)

        # 现在通过映射的原理，去掉重复的城市点
        temporary1 = []
        temporary2 = []
        for i in range(len(new_chrom1)):
            if new_chrom1[i] not in temporary1:
                temporary1.append(new_chrom1[i])
        for i in range(len(new_chrom2)):
            if new_chrom2[i] not in temporary2:
                temporary2.append(new_chrom2[i])
        chrom1 = temporary1
        chrom2 = temporary2
        return chrom1, chrom2
    else:
        return parent1[:], parent2[:]  # 输出两个进行杂交操作后的两个个体


def mutate(chrom):  # 变异函数
    if random.random() < pm:  # 按照一定的概率进行变异操作
        mpoint1 = random.randint(0, city_number - 1)  # 随机产生两个变异位置
        mpoint2 = random.randint(0, city_number - 1)
        # 交换变异点的基因位
        temp = chrom[mpoint1]
        chrom[mpoint1] = chrom[mpoint2]
        chrom[mpoint2] = temp
    return chrom

def distance_value(distance_matrix, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i]][path[i + 1]]
    return distance

args = arg_parser()
lines = open(args.file, 'r').readlines()

# 终止条件：最大迭代次数
N = args.N
# 遗传算法参数
pop_size = 100  # 种群数
pc = 0.9  # 交叉概率
pm = 0.05  # 突变概率

gap = []
length = []
begin = time.time()
for line in tqdm(lines[:5]):
    line = line.strip()

    # Extract points
    points = line.split(" output ")[0]
    points = points.split(" ")
    points = np.array(
        [[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)]
    )

    dis_mat = (((points.reshape(points.shape[0], 1, 2) - points.reshape(1, points.shape[0], 2)) ** 2).sum(-1)) ** 0.5

    # Extract tour
    gt_tour = line.split(" output ")[1]
    gt_tour = gt_tour.split(" ")
    gt_tour = [int(t) - 1 for t in gt_tour]

    city_number = len(gt_tour) - 1

    # 主程序
    best_path = []
    best_fitness = 0.0
    Best_Fitness = []  # 用来存放每代种群中最个体的适应值
    population = initpopulation(pop_size, city_number)  # 初始种群
    fit_array = fitness(population, dis_mat)

    for iter in range(N):
        iter += 1
        fit_array = fitness(population, dis_mat)  # 适应值列表
        max_fitness = max(fit_array)
        max_index = fit_array.index(max_fitness)
        lx = []
        ly = []
        for i in population[max_index][:]:
            i = int(i)
            lx.append(points[i][0])
            ly.append(points[i][1])

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_path = population[max_index][:]
            x = copy.copy(lx)
            y = copy.copy(ly)

        Best_Fitness.append(best_fitness)

        new_population = []
        n = 0
        while n < pop_size:
            p1 = select(population, fit_array)
            p2 = select(population, fit_array)
            while p2 == p1:
                p2 = select(population, fit_array)
            # 交叉
            chrom1, chrom2 = crossover(p1, p2)
            # 变异
            chrom1 = mutate(chrom1)
            chrom2 = mutate(chrom2)
            new_population.append(chrom1)
            new_population.append(chrom2)
            n += 2
        population = new_population

        last_best_fitness = 0
        if last_best_fitness < math.sqrt(1 / best_fitness):
            last_best_fitness = math.sqrt(1 / best_fitness)
            last_best_path = best_path

    ga_path =  best_path + [best_path[0]]
    gt_length = distance_value(dis_mat, gt_tour)
    ga_length = distance_value(dis_mat, ga_path)

    gap.append((ga_length - gt_length) / gt_length)
    length.append(ga_length)

end=time.time()
print(np.array(length).mean())
print(np.array(gap).mean())
print(end - begin)
