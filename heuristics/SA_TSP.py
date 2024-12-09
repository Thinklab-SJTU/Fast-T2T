import random
import math
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import time


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--loop", type=int, default=10)

    args = parser.parse_args()
    return args


class SA_TSP():
    def __init__(self, distance_matrix, innerLoop=10):
        self.T0 = 4000  # 以学号为初始温度
        self.distance_matrix = distance_matrix
        n = len(distance_matrix)
        self.initial0 = list(range(n)) + [0]
        self.innerLoop = innerLoop

    def update_T(self, T):
        T = T * 0.99
        return T

    def Metropolis(self, value1, value2, T):
        return math.exp((value1 - value2) / T)

    def swap_city(self, old_path):
        new_path = old_path.copy()
        city1 = random.randint(1, len(old_path) - 2)
        city2 = random.randint(1, len(old_path) - 2)

        if city1 != city2:
            temp = new_path[city1]
            new_path[city1] = new_path[city2]
            new_path[city2] = temp
        else:
            new_path = self.swap_city(old_path)
        return new_path

    def distance_value(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i + 1]]
        return distance

    def SA_alogo(self):
        # result = []
        T = self.T0
        T_f = T * (0.4 ** 12)
        initial_solution = self.initial0
        while T > T_f:
            index = self.innerLoop
            for _ in range(index):
                other_solution = self.swap_city(initial_solution)
                value1 = self.distance_value(initial_solution)
                value2 = self.distance_value(other_solution)
                pob = random.uniform(0, 1)
                if value1 > value2:
                    initial_solution = other_solution
                elif self.Metropolis(value1, value2, T) > pob:
                    # print(str(self.Metropolis(value1,value2,T)))
                    initial_solution = other_solution
            T = self.update_T(T)
        result = initial_solution
        # print(str(result))
        return result

    def test_swap_city(self):
        new_path = self.swap_city(self.initial0)
        print(str(new_path))

    def test_distance_compute(self):
        distance_of_path = self.distance_value(self.initial0)
        print(distance_of_path)


if __name__ == '__main__':
    random.seed(28)
    args = arg_parser()
    file = args.file
    loop = args.loop
    lines = open(file, 'r').readlines()
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

        adj_matrix = (((points.reshape(points.shape[0], 1, 2) - points.reshape(1, points.shape[0], 2)) ** 2).sum(-1)) ** 0.5

        # Extract tour
        gt_tour = line.split(" output ")[1]
        gt_tour = gt_tour.split(" ")
        gt_tour = [int(t) - 1 for t in gt_tour]

        solution = SA_TSP(adj_matrix, loop)
        gt_distance = solution.distance_value(gt_tour)
        # solution.test_swap_city()
        # solution.test_distance_compute()
        result = solution.SA_alogo()
        result_distance = solution.distance_value(result)

        gap.append((result_distance - gt_distance) / gt_distance)
        length.append(result_distance)

    end = time.time()
    print(np.array(length).mean())
    print(np.array(gap).mean())
    print(end - begin)
