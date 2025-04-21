import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArrayTransducer:
    def __init__(self, num_elements, element_pitch, element_width, Lx, Ly):
        self.num_elements = num_elements  # 阵元数量
        self.element_pitch = element_pitch  # 阵元间距
        self.element_width = element_width  # 阵元宽度
        self.Lx = Lx  # 仿真区域宽度
        self.Ly = Ly  # 仿真区域高度
        self.positions = np.zeros((num_elements, 2))  # 阵元位置 (x, y)
        self.delays = np.zeros(num_elements)  # 各阵元延迟时间
        self._initialize_positions()

    def _initialize_positions(self):
        """初始化阵元位置"""
        total_width = (self.num_elements - 1) * self.element_pitch
        start_x = (self.Lx - total_width) / 2  # 水平居中
        y_pos = self.Ly - 0.001  # 距上边界0.001米
        for i in range(self.num_elements):
            self.positions[i] = [start_x + i * self.element_pitch, y_pos]

    def calculate_delays(self, focus_point):
        """计算聚焦延迟，考虑阵元方向性和加权"""
        c = 1500.0  # 声速 (m/s)
        wavelength = c / 2.5e6  # 波长 (对应2.5MHz频率)
        k = 2 * np.pi / wavelength  # 波数
        
        # 计算每个阵元到焦点的距离和角度
        distances = np.sqrt(np.sum((self.positions - focus_point)**2, axis=1))
        angles = np.arctan2(focus_point[1] - self.positions[:, 1],
                           focus_point[0] - self.positions[:, 0])
        
        # 计算方向性因子 (使用简化的方向性模型)
        directivity = np.sinc(k * self.element_width * np.sin(angles) / 2)
        
        # 计算Hanning窗加权系数（用于抑制旁瓣）
        weights = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.num_elements) / 
                                  (self.num_elements - 1)))
        
        # 计算延迟时间（考虑方向性和加权）
        max_distance = np.max(distances)
        self.delays = (max_distance - distances) / c
        
        # 存储方向性和加权系数供激励使用
        self.directivity = directivity
        self.weights = weights
        
        return self.delays

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

class BeamformingSimulator:
    def __init__(self, parameter, array_transducer, emission_mode='single'):
        # 仿真参数
        self.parameter = parameter
        self.array_transducer = array_transducer
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
        
        # 发射模式设置
        self.emission_mode = emission_mode  # 'single'为单次发射，'continuous'为连续发射
        self.source_active = True
        self.source = self._generate_source()
        self.source_period = len(self.source)  # 声源周期
        
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

        # 施加延迟声源
        if self.source_active:
            # 根据发射模式处理声源
            if self.emission_mode == 'continuous':
                current_frame = frame % self.source_period
            else:  # 单次发射模式
                current_frame = frame
                if frame >= self.source_period:
                    self.source_active = False
                    current_frame = -1  # 确保不会激发声源
            
            if current_frame >= 0:
                for i, pos in enumerate(self.array_transducer.positions):
                    # 计算延迟后的帧索引
                    delayed_frame = current_frame - int(self.array_transducer.delays[i] / self.dt)
                    if delayed_frame >= 0 and delayed_frame < self.source_period:
                        x = int(pos[0] / self.parameter.dx)
                        y = int(pos[1] / self.parameter.dy)
                        # 应用方向性和加权系数
                        amplitude = self.source[delayed_frame] * \
                                   self.array_transducer.directivity[i] * \
                                   self.array_transducer.weights[i]
                        self.u_next[x, y] += amplitude

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
        plt.title('Ultrasound Beamforming Simulation')

        # 绘制换能器阵元位置
        plt.scatter(simulation.array_transducer.positions[:, 0],
                   simulation.array_transducer.positions[:, 1],
                   c='white', marker='s', s=50, label='Transducer Elements')
        plt.legend()

    def update(self, frame):
        """更新动画"""
        self.simulation.update(frame)
        self.im.set_data(self.simulation.u_curr.T)
        return [self.im]

def main():
    # 创建参数对象
    parameter = Parameter(Lx=0.02, Ly=0.02, nx=200, ny=200)
    
    # 创建换能器阵列
    array_transducer = ArrayTransducer(
        num_elements=32,  # 32个阵元
        element_pitch=0.0002,  # 阵元间距0.2mm
        element_width=0.0001,  # 阵元宽度0.1mm
        Lx=parameter.Lx,  # 仿真区域宽度
        Ly=parameter.Ly  # 仿真区域高度
    )
    
    # 设置聚焦点
    focus_point = np.array([0.01, 0.01])  # 聚焦点位置在阵列下方中心区域
    array_transducer.calculate_delays(focus_point)

    # 选择发射模式 ('single'为单次发射，'continuous'为连续发射)
    emission_mode = input("请选择发射模式 (0为单次发射，1为连续发射): ") or '0'
    emission_mode = 'single' if emission_mode == '0' else 'continuous' 
     # 可以修改为'continuous'进行连续发射

    # 创建仿真对象
    simulation = BeamformingSimulator(parameter, array_transducer, emission_mode=emission_mode)

    # 创建显示对象
    display = Display(simulation)

    # 创建动画
    ani = FuncAnimation(display.fig, display.update,
                       frames=simulation.nt, interval=1, blit=True)
    plt.show()

if __name__ == '__main__':
    main()