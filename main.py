import numpy as np
import matplotlib.pyplot as plt

# 定义城市的位置，这里使用随机生成的点作为示例
np.random.seed(0)  # 为了结果的可重复性
num_cities =5
city_positions = np.random.rand(num_cities, 2) * 100  # 假设城市在100x100的平面上

# 计算城市之间的距离矩阵
def distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[j, i] = dist_matrix[i, j]  # 距离矩阵是对称的
    return dist_matrix

# 计算能量函数
def energy_function(cities, tour):
    num_cities = len(cities)
    energy = 0
    for i in range(num_cities):
        energy += -distance_matrix(cities)[tour[i], tour[(i+1) % num_cities]]
    return energy

# 更新规则
def update_tour(tour, cities):
    num_cities = len(cities)
    for i in range(num_cities):
        for j in range(i + 2, num_cities):
            if energy_function(cities, tour) + distance_matrix(cities)[tour[i], tour[j]] > \
               energy_function(cities, tour[:i] + tour[i+1:j] + [tour[i], tour[j]] + tour[j+1:]):
                tour = tour[:i] + [tour[i], tour[j]
                                   ] + tour[i + 1:j] + tour[j + 1:]
    return tour

# 初始化路径
tour = list(range(num_cities))
np.random.shuffle(tour)
energy_list=[]

# 迭代更新路径直到收敛
max_iterations = 1000
for _ in range(max_iterations):
    energy_list.append(energy_function(city_positions, tour))
    new_tour = update_tour(tour, city_positions)
    if np.array_equal(tour, new_tour):
        break
    tour = new_tour
# 输出路径顺序
print("Path order:", tour)

# 绘制城市位置和路径
plt.figure(figsize=(8, 8))
plt.scatter(city_positions[:, 0], city_positions[:, 1], c='red', marker='o', label='Cities')
for i in range(num_cities):
    start = city_positions[tour[i]]
    end = city_positions[tour[(i+1) % num_cities]]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')
plt.title('TSP Solution with Continuous HNN')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.savefig("main.png")
