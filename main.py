import numpy as np
import matplotlib.pyplot as plt
import random

from src.hnn import HNN

def get_distance(point1,point2):
    """
    计算两点之间的欧几里得距离。
    """
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return float(distance)

def main():
    # 任取五个点
    points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(5)]
    print(f"五个点的坐标为{points}")

    # 省略

    # 可视化
    plt.figure(figsize=(10, 10))
    for point in points:
        plt.plot(point[0], point[1], 'ro')
    #plt.axis('off')
    plt.savefig('plot.png', bbox_inches='tight', pad_inches=0)#


if __name__ == "__main__":
    main()