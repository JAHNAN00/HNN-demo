import numpy as np
import matplotlib.pyplot as plt
import itertools

class HNN:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        np.fill_diagonal(self.weights, 0)

    def train(self, patterns):
        """
        使用Hebbian学习规则训练网络，存储多个模式
        :param patterns: 包含多个模式的列表，每个模式是一个一维的numpy数组
        """
        for p in patterns:
            p = np.array(p)
            self.weights += np.outer(p, p)  # Hebbian学习规则
        self.weights = np.tril(self.weights) + np.tril(self.weights, -1).T

    def run(self, input_pattern, max_iterations=100):
        """
        输入一个模式，进行状态更新，直到网络收敛
        :param input_pattern: 输入模式（一维numpy数组）
        :param max_iterations: 最大迭代次数
        :return: 收敛后的模式
        """
        state = np.copy(input_pattern)
        for _ in range(max_iterations):
            prev_state = np.copy(state)
            for i in range(self.n_neurons):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1  # 激活函数：符号函数
            if np.array_equal(state, prev_state):
                break
        return state

def generate_cities(n):
    """
    随机生成城市坐标
    :param n: 城市数量
    :return: 城市的坐标（numpy数组）
    """
    return np.random.rand(n, 2)

def calculate_distances(cities):
    """
    计算城市之间的距离矩阵
    :param cities: 城市坐标
    :return: 距离矩阵
    """
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(cities[i] - cities[j])  # 欧几里得距离
            distances[i, j] = distance
            distances[j, i] = distance
    return distances

def tsp_energy_function(state, distances):
    """
    计算给定状态下的能量函数值
    :param state: 神经元状态
    :param distances: 城市之间的距离矩阵
    :return: 能量值
    """
    energy = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == 1 and state[j] == 1:  # 计算这两城市是否在路径中
                energy += distances[i, j]  # 加上路径长度
    return energy

def main():
    n_cities = 4
    # 生成城市和距离
    #cities = generate_cities(n_cities)
    cities=np.array([[0,0],[0,1],[1,1],[1,0]])
    distances = calculate_distances(cities)

    # 假设每条路径对应一个神经元
    # 创建一个n_cities * n_cities的矩阵，表示每个城市之间是否有路径
    n_neurons = 16
    hnn = HNN(n_neurons)

    # 假设初始化模式为随机状态
    initial_state = np.random.choice([-1, 1], size=(n_cities, n_cities))

    # 训练HNN网络
    hnn.train([initial_state.flatten()])

    # 优化状态
    optimized_state = hnn.run(initial_state.flatten())

    # 解析优化结果，获得旅行路径
    path = []
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            if optimized_state[i * (n_cities - 1) // 2 + j] == 1:
                path.append((i, j))
    print(path)

    # 可视化路径
    plt.figure(figsize=(8, 8))
    plt.scatter(cities[:, 0], cities[:, 1], color='red')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], f'City {i+1}', fontsize=12, ha='right')
    
    for (i, j) in path:
        plt.plot([cities[i, 0], cities[j, 0]], [cities[i, 1], cities[j, 1]], 'b-', lw=2)

    plt.title('TSP Solution (Optimized Path)')
    plt.savefig('plot.png')

if __name__ == "__main__":
    main()
