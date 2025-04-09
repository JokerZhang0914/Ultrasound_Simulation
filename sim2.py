import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 仿真参数设置
Lx = 0.02        # x方向尺寸 (m)
Ly = 0.02        # y方向尺寸 (m)
nx = 200         # x方向网格数
ny = 200         # y方向网格数
dx = Lx / nx     # x方向步长
dy = Ly / ny     # y方向步长

start_pos_x1 = 0.015  # 第一个区域起始位置x (m)
end_pos_x1 = 0.018   # 第一个区域结束位置x (m)
start_pos_y1 = 0 # 第一个区域起始位置y (m)
end_pos_y1 = 0.02   # 第一个区域结束位置y (m)
ratio1 = 0.6      # 第一个区域声阻抗与基础声阻抗的比值

# 声速场设置（二维数组）
c0 = 1500.0        # 基础声速 (m/s)
c = np.full((nx, ny), c0)  # 基础声速
# 设置矩形区域（0.03-0.04m范围）的声速变化
x_start, x_end = int(start_pos_x1/dx), int(end_pos_x1/dx)
y_start, y_end = int(start_pos_y1/dy), int(end_pos_y1/dy)
c[x_start:x_end, y_start:y_end] = c0 * ratio1  # 修改区域声速

f0 = 2.5e6        # 中心频率 (Hz)
cycles = 4        # 周期数
t_end = 25e-6     # 总时间 (s)
CFL = 0.4         # 更严格的CFL条件系数（二维稳定性要求）

# 时间步长计算（基于最大声速）
dt = CFL * min(dx, dy) / (np.sqrt(2)*np.max(c))
nt = int(t_end / dt)

# 初始化波场（二维数组）
u_curr = np.zeros((nx, ny))  # 当前时间步
u_prev = u_curr.copy()        # 前一时间步
u_next = u_curr.copy()        # 下一时间步

# 生成二维声源（中心点声源）
source_pos1 = (0, 40)  
source_pos2 = (160, 160)
num_source_points = int(round(cycles/(f0*dt)))
t_source = np.linspace(0, cycles/f0, num_source_points)
source = np.exp(-(t_source - 0.5*t_source[-1])**2/((t_source[-1]/4)**2)) * np.sin(2*np.pi*f0*t_source)
source_active = True

# 创建绘图对象
fig = plt.figure(figsize=(8,6))
im = plt.imshow(u_curr.T, cmap='coolwarm', origin='lower', 
                extent=[0, Lx, 0, Ly], vmin=-0.5, vmax=0.5)
plt.colorbar(label='Wave Amplitude')
plt.title('2D Ultrasound Wave Propagation')

def update(frame):
    global u_prev, u_curr, source_active
    
    # 二维波动方程计算（不含边界）
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            laplacian = (u_curr[i+1,j] - 2*u_curr[i,j] + u_curr[i-1,j])/dx**2 \
                      + (u_curr[i,j+1] - 2*u_curr[i,j] + u_curr[i,j-1])/dy**2
            u_next[i,j] = 2*u_curr[i,j] - u_prev[i,j] + (c[i,j]*dt)**2 * laplacian

    # 吸收边界条件（二维Mur边界）
    alpha = 0.0
    # 四周边界
    u_next[0, :] = alpha * u_curr[1, :]    # 左边界
    u_next[-1, :] = alpha * u_curr[-2, :]  # 右边界
    u_next[:, 0] = alpha * u_curr[:, 1]    # 下边界
    u_next[:, -1] = alpha * u_curr[:, -2]  # 上边界

    # 施加声源（中心点声源）
    if frame < len(source) and source_active:
        u_next[source_pos1[0], source_pos1[1]] += source[frame]
        u_next[source_pos2[0], source_pos2[1]] += source[frame]
    
    if frame == len(source) - 1:
        source_active = False

    # 更新波场
    u_prev[:] = u_curr
    u_curr[:] = u_next
    
    im.set_data(u_curr.T)
    return [im]

# 调整interval参数确保实时显示
ani = FuncAnimation(fig, update, frames=nt, interval=1, blit=True)
plt.show()