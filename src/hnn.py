import numpy as np


class HNN:

    def __init__(self, n_neurons):
        """
        初始化Hopfield神经网络
        :param n_neurons: 网络中神经元的数量
        """
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))  # 权重矩阵
        np.fill_diagonal(self.weights, 0)  # 自连接权重为0

    def train(self, patterns):
        """
        使用Hebbian学习规则训练网络，存储多个模式
        :param patterns: 包含多个模式的列表，每个模式是一个一维的numpy数组
        """
        for p in patterns:
            p = np.array(p)
            self.weights += np.outer(p, p)  # Hebbian学习规则
        # 保证权重矩阵对称
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
            # 异步更新：逐个神经元更新状态
            for i in range(self.n_neurons):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1  # 激活函数：符号函数
            # 如果状态没有变化，表示已经收敛
            if np.array_equal(state, prev_state):
                break
        return state

if __name__ == "__main__":
    # 定义两个简单的模式（+1/-1 二值模式）
    pattern1 = np.array([1, -1, 1, -1])
    pattern2 = np.array([-1, 1, -1, 1])

    # 创建一个HNN实例，假设有4个神经元
    hnn = HNN(n_neurons=4)

    # 训练网络，存储模式
    hnn.train([pattern1, pattern2])

    # 测试1：用一个完整的模式来恢复存储的模式
    input_pattern = np.array([1, -1, 1, -1])  # 输入模式与pattern1一致
    print("Test 1: Complete input pattern")
    restored_pattern = hnn.run(input_pattern)
    print("Restored pattern:", restored_pattern)

    # 测试2：用部分的模式来恢复存储的模式
    input_pattern = np.array([1, -1, 1, 1])  # 输入模式部分失真
    print("\nTest 2: Incomplete input pattern")
    restored_pattern = hnn.run(input_pattern)
    print("Restored pattern:", restored_pattern)

    # 测试3：用带噪声的模式来恢复存储的模式
    input_pattern = np.array([1, 1, -1, -1])  # 输入模式带有噪声
    print("\nTest 3: Noisy input pattern")
    restored_pattern = hnn.run(input_pattern)
    print("Restored pattern:", restored_pattern)
