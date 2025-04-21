import numpy as np
import matplotlib.pyplot as plt
def das_beamforming(transducer_positions, signals, grid, sound_speed):
    """
    实现延迟叠加算法的超声成像。

    参数:
    - transducer_positions: 换能器阵列的横向坐标 (N,)
    - signals: 每个换能器接收到的信号 (N, T)
    - grid: 成像区域网格 (Mx, Mz, 2)
    - sound_speed: 介质中的声速

    返回:
    - image: 成像区域的聚焦信号强度 (Mx, Mz)
    """
    Mx, Mz, _ = grid.shape
    N, T = signals.shape
    image = np.zeros((Mx, Mz))

    for ix in range(Mx):
        for iz in range(Mz):
            x, z = grid[ix, iz]
            signal_sum = 0
            for i in range(N):
                # 计算到成像点的延迟时间
                delay = np.sqrt((x - transducer_positions[i])**2 + z**2) / sound_speed
                # 转换为采样点索引
                idx = int(delay * T)
                if idx < T:
                    signal_sum += signals[i, idx]
            image[ix, iz] = signal_sum
    return image
    # 模拟换能器阵列
N = 8  # 换能器个数
T = 1000  # 信号长度
c = 1500  # 声速 (m/s)
transducer_positions = np.linspace(-0.05, 0.05, N)  # 换能器横向位置 (m)

# 模拟信号
signals = np.random.rand(N, T)

# 创建成像网格
Mx, Mz = 100, 100  # 网格尺寸
x = np.linspace(-0.05, 0.05, Mx)
z = np.linspace(0.01, 0.1, Mz)
grid = np.array([[(xi, zi) for zi in z] for xi in x])

# 应用DAS算法
image = das_beamforming(transducer_positions, signals, grid, c)

# 可视化成像结果
plt.imshow(image, extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='hot')
plt.colorbar(label="Amplitude")
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("DAS Beamformed Image")
plt.show()