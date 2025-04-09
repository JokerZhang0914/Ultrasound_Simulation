import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 仿真参数设置
L = 0.05           # 介质长度 (m)
nx = 1500         # 空间点数
dx = L / nx       # 空间步长
c = 1500.0        # 声速 (m/s)
f0 = 2.5e6        # 中心频率 (Hz)
cycles = 5        # 周期数
t_end = 25e-6     # 总时间 (s)
CFL = 0.5         # CFL条件系数

# 计算时间步长
dt = CFL * dx / c
nt = int(t_end / dt)

# 初始化波场
u_curr = np.zeros(nx)  # 当前时间步
u_prev = np.zeros(nx)  # 前一时间步
u_next = np.zeros(nx)  # 下一时间步

# 生成衰减正弦波声源
num_source_points = int(round(cycles/(f0*dt)))
t_source = np.linspace(0, cycles/f0, num_source_points)
source = np.exp(-(t_source - 0.5*t_source[-1])**2/((t_source[-1]/4)**2)) * np.sin(2*np.pi*f0*t_source)
source_active = True

# 边界条件参数（左边界吸收，右边界全反射）
alpha = 0.9  # 左边界吸收系数

# 创建绘图对象
fig, ax = plt.subplots()
line, = ax.plot(np.linspace(0, L, nx), u_curr)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Position (m)')
ax.set_title('Ultrasound Wave Propagation')

def update(frame):
    global u_prev, u_curr,source_active
    
    # 更新波场内部节点
    u_next[1:-1] = 2*u_curr[1:-1] - u_prev[1:-1] + (c*dt/dx)**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
    
    # 应用边界条件（左边界吸收，右边界全反射）
    u_next[0] = u_curr[1]  # 左边界吸收
    u_next[-1] = 0 * u_curr[-2]  # 右边界全反射（Neumann边界条件）
    
    # 施加声源（仅在初始时间步施加）
    if frame < len(source) and source_active:
        u_next[0] = source[frame]  # 覆盖边界条件设置的值
    
    if frame == len(source) - 1:
        source_active = False # 标记声源结束,防止repeat后重复施加

    # 更新波场
    u_prev[:] = u_curr
    u_curr[:] = u_next
    
    line.set_ydata(u_curr)
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=0.1, blit=True, repeat = True)
plt.show()