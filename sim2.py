import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Parameter:
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.base_speed = 1500.0  # 基础声速 (m/s)
        self.regions = []  # 存储不同声阻抗区域的列表
        self.sources = []  # 存储声源位置的列表

    def add_source(self, pos_x, pos_y):
        """添加一个新的声源位置"""
        self.sources.append({
            'pos_x': pos_x,
            'pos_y': pos_y
        })

    def add_region(self, start_pos_x, end_pos_x, start_pos_y, end_pos_y, ratio):
        """添加一个新的声阻抗区域"""
        self.regions.append({
            'start_pos_x': start_pos_x,
            'end_pos_x': end_pos_x,
            'start_pos_y': start_pos_y,
            'end_pos_y': end_pos_y,
            'ratio': ratio
        })

    def get_speed(self):
        """生成声速分布数组"""
        c = np.full((self.nx, self.ny), self.base_speed)
        for region in self.regions:
            x_start = int(region['start_pos_x'] / self.dx)
            x_end = int(region['end_pos_x'] / self.dx)
            y_start = int(region['start_pos_y'] / self.dy)
            y_end = int(region['end_pos_y'] / self.dy)
            c[x_start:x_end, y_start:y_end] = self.base_speed * region['ratio']
        return c

class Simulator:
    def __init__(self, parameter):
        # 仿真参数
        self.parameter = parameter
        self.f0 = 2.5e6  # 中心频率 (Hz)
        self.cycles = 4  # 周期数
        self.t_end = 25e-6  # 总时间 (s)
        self.CFL = 0.4  # CFL条件系数（二维稳定性要求）
        
        # 初始化声速场
        self.c = parameter.get_speed()
        
        # 计算时间步长
        self.dt = self.CFL * min(parameter.dx, parameter.dy) / (np.sqrt(2)*np.max(self.c))
        self.nt = int(self.t_end / self.dt)
        
        # 初始化波场
        self.u_curr = np.zeros((parameter.nx, parameter.ny))
        self.u_prev = self.u_curr.copy()
        self.u_next = self.u_curr.copy()
        
        # 生成声源
        self.source_active = True
        self.source = self._generate_source()
        
        # 获取声源位置列表
        self.sources = parameter.sources
        
        # 边界条件参数
        self.c_max = np.max(self.c)  # 最大声速，用于边界条件计算

    def _generate_source(self):
        """生成衰减正弦波声源"""
        num_source_points = int(round(self.cycles/(self.f0*self.dt)))
        t_source = np.linspace(0, self.cycles/self.f0, num_source_points)
        return np.exp(-(t_source - 0.5*t_source[-1])**2/((t_source[-1]/4)**2)) * \
               np.sin(2*np.pi*self.f0*t_source)

    def update(self, frame):
        """更新波场"""
        # 二维波动方程计算（不含边界）
        for i in range(1, self.parameter.nx-1):
            for j in range(1, self.parameter.ny-1):
                laplacian = (self.u_curr[i+1,j] - 2*self.u_curr[i,j] + self.u_curr[i-1,j])/self.parameter.dx**2 \
                          + (self.u_curr[i,j+1] - 2*self.u_curr[i,j] + self.u_curr[i,j-1])/self.parameter.dy**2
                self.u_next[i,j] = 2*self.u_curr[i,j] - self.u_prev[i,j] + (self.c[i,j]*self.dt)**2 * laplacian

        # 全透射边界条件（一阶Mur边界）
        # 左边界
        self.u_next[0, 1:-1] = self.u_curr[1, 1:-1] + \
                             (self.c_max*self.dt - self.parameter.dx)/(self.c_max*self.dt + self.parameter.dx) * \
                             (self.u_next[1, 1:-1] - self.u_curr[0, 1:-1])
        
        # 右边界
        self.u_next[-1, 1:-1] = self.u_curr[-2, 1:-1] + \
                              (self.c_max*self.dt - self.parameter.dx)/(self.c_max*self.dt + self.parameter.dx) * \
                              (self.u_next[-2, 1:-1] - self.u_curr[-1, 1:-1])
        
        # 下边界
        self.u_next[1:-1, 0] = self.u_curr[1:-1, 1] + \
                             (self.c_max*self.dt - self.parameter.dy)/(self.c_max*self.dt + self.parameter.dy) * \
                             (self.u_next[1:-1, 1] - self.u_curr[1:-1, 0])
        
        # 上边界
        self.u_next[1:-1, -1] = self.u_curr[1:-1, -2] + \
                              (self.c_max*self.dt - self.parameter.dy)/(self.c_max*self.dt + self.parameter.dy) * \
                              (self.u_next[1:-1, -2] - self.u_curr[1:-1, -1])
        
        # 处理四个角落点
        # 左下角
        self.u_next[0, 0] = self.u_curr[1, 1] + \
                         (self.c_max*self.dt - np.sqrt(self.parameter.dx**2 + self.parameter.dy**2))/ \
                         (self.c_max*self.dt + np.sqrt(self.parameter.dx**2 + self.parameter.dy**2)) * \
                         (self.u_next[1, 1] - self.u_curr[0, 0])
        
        # 右下角
        self.u_next[-1, 0] = self.u_curr[-2, 1] + \
                          (self.c_max*self.dt - np.sqrt(self.parameter.dx**2 + self.parameter.dy**2))/ \
                          (self.c_max*self.dt + np.sqrt(self.parameter.dx**2 + self.parameter.dy**2)) * \
                          (self.u_next[-2, 1] - self.u_curr[-1, 0])
        
        # 左上角
        self.u_next[0, -1] = self.u_curr[1, -2] + \
                          (self.c_max*self.dt - np.sqrt(self.parameter.dx**2 + self.parameter.dy**2))/ \
                          (self.c_max*self.dt + np.sqrt(self.parameter.dx**2 + self.parameter.dy**2)) * \
                          (self.u_next[1, -2] - self.u_curr[0, -1])
        
        # 右上角
        self.u_next[-1, -1] = self.u_curr[-2, -2] + \
                           (self.c_max*self.dt - np.sqrt(self.parameter.dx**2 + self.parameter.dy**2))/ \
                           (self.c_max*self.dt + np.sqrt(self.parameter.dx**2 + self.parameter.dy**2)) * \
                           (self.u_next[-2, -2] - self.u_curr[-1, -1])

        # 施加声源
        if frame < len(self.source) and self.source_active:
            for source in self.sources:
                x = int(source['pos_x'] / self.parameter.dx)
                y = int(source['pos_y'] / self.parameter.dy)
                self.u_next[x, y] += self.source[frame]
        
        if frame == len(self.source) - 1:
            self.source_active = False

        # 更新波场
        self.u_prev[:] = self.u_curr
        self.u_curr[:] = self.u_next

class Display:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig = plt.figure(figsize=(8,6))
        self.im = plt.imshow(simulation.u_curr.T, cmap='coolwarm', origin='lower',
                            extent=[0, simulation.parameter.Lx, 0, simulation.parameter.Ly],
                            vmin=-0.5, vmax=0.5)
        plt.colorbar(label='Wave Amplitude')
        plt.title('2D Ultrasound Wave Propagation')

    def update(self, frame):
        """更新动画"""
        self.simulation.update(frame)
        self.im.set_data(self.simulation.u_curr.T)
        return [self.im]

def main():
    # 创建参数对象
    parameter = Parameter(Lx=0.02, Ly=0.02, nx=200, ny=200)
    
    # 添加声阻抗区域
    parameter.add_region(start_pos_x=0.015, end_pos_x=0.018,
                        start_pos_y=0, end_pos_y=0.02, ratio=0.6)
    
    # 添加多个声源
    parameter.add_source(pos_x=0.002, pos_y=0.004)  # 左下角声源
    parameter.add_source(pos_x=0.016, pos_y=0.016)  # 右上角声源
    parameter.add_source(pos_x=0.01, pos_y=0.01)    # 中心声源

    # 创建仿真对象
    simulation = Simulator(parameter)

    # 创建显示对象
    display = Display(simulation)

    # 创建动画
    ani = FuncAnimation(display.fig, display.update,
                       frames=simulation.nt, interval=1, blit=True)
    plt.show()

if __name__ == '__main__':
    main()