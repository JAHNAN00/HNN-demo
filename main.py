import numpy as np
import matplotlib.pyplot as plt

from src.hnn import HNN

def generate_pattern(n):
    """
    随机生成一个二进制模式，-1 和 1 表示神经元的两种状态
    :param n: 模式的长度
    :return: 生成的模式（numpy数组）
    """
    return np.random.choice([-1, 1], size=n)

def main():
    # 设置神经元数量
    n_neurons = 10
    
    # 生成训练模式
    patterns = [generate_pattern(n_neurons) for _ in range(3)]  # 生成3个随机模式

    # 创建并训练HNN
    hnn = HNN(n_neurons)
    hnn.train(patterns)
    
    # 生成一个测试模式（可以是一个稍微改变的模式）
    test_pattern = patterns[0].copy()
    test_pattern[2] = -test_pattern[2]  # 修改一个元素来模拟噪声

    # 运行网络，恢复最接近的模式
    result = hnn.run(test_pattern)
    
    print("原始模式:")
    print(patterns[0])
    print("测试模式（带噪声）:")
    print(test_pattern)
    print("恢复的模式:")
    print(result)

    # 可视化
    plt.subplot(1, 3, 1)
    plt.title("Original Pattern")
    plt.imshow(patterns[0].reshape(1, -1), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Test Pattern")
    plt.imshow(test_pattern.reshape(1, -1), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Recovered Pattern")
    plt.imshow(result.reshape(1, -1), cmap="gray")
    plt.savefig('plot.png')
    
if __name__ == "__main__":
    main()