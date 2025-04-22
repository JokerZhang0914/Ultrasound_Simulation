import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArrayTransducer:
    def __init__(self, num_elements, element_pitch, element_width, Lx, Ly, array_type='linear', radius=None):
        self.num_elements = num_elements  # 阵元数量
        self.element_pitch = element_pitch  # 阵元间距
        self.element_width = element_width  # 阵元宽度
        self.Lx = Lx  # 仿真区域宽度
        self.Ly = Ly  # 仿真区域高度
        self.array_type = array_type  # 阵列类型：'linear'或'curved'
        self.radius = radius  # 弧形阵列的曲率半径
        self.positions = np.zeros((num_elements, 2))  # 阵元位置 (x, y)
        self.delays = np.zeros(num_elements)  # 各阵元延迟时间
        self._initialize_positions()

    def _initialize_positions(self):
        """初始化阵元位置"""
        total_width = (self.num_elements - 1) * self.element_pitch
        start_x = (self.Lx - total_width) / 2  # 水平居中
        y_pos = self.Ly - 0.001  # 距上边界0.001米
        
        if self.array_type == 'linear':
            # 线性阵列
            for i in range(self.num_elements):
                self.positions[i] = [start_x + i * self.element_pitch, y_pos]
        else:  # 弧形阵列
            # 初始化中心阵元位置
            center_element_idx = self.num_elements // 2
            center_x = self.Lx / 2
            center_y = y_pos
            self.positions[center_element_idx] = [center_x, center_y]
            
            # 计算弧形阵列的参数
            total_angle = 2 * np.arcsin(total_width / (2 * total_width * 2))  # 使用固定的弧度范围
            angle_step = total_angle / (self.num_elements - 1)
            start_angle = -total_angle / 2
            
            # 计算其他阵元的位置
            for i in range(self.num_elements):
                if i != center_element_idx:
                    angle = start_angle + i * angle_step
                    x = center_x + total_width * np.sin(angle)
                    y = center_y + total_width * (1 - np.cos(angle))
                    self.positions[i] = [x, y]

    def calculate_delays(self, focus_point):
        """计算聚焦延迟，考虑阵元方向性和加权，并动态调整弧形阵列位置"""
        if self.array_type == 'curved':
            # 获取中心阵元位置
            center_element_idx = self.num_elements // 2
            center_pos = self.positions[center_element_idx].copy()
            
            # 计算从中心阵元到聚焦点的方向向量
            direction = focus_point - center_pos
            direction_norm = np.sqrt(np.sum(direction**2))
            direction = direction / direction_norm
            
            # 计算圆心位置（在中心阵元背后）
            total_width = (self.num_elements - 1) * self.element_pitch
            radius = total_width * 2  # 使用固定的半径
            circle_center = center_pos - direction * radius
            
            # 计算旋转角度范围
            total_angle = 2 * np.arcsin(total_width / (2 * radius))
            angle_step = total_angle / (self.num_elements - 1)
            start_angle = -total_angle / 2
            
            # 计算旋转轴（垂直于方向向量的单位向量）
            rotation_axis = np.array([-direction[1], direction[0]])
            
            # 更新所有阵元位置
            for i in range(self.num_elements):
                if i != center_element_idx:
                    angle = start_angle + i * angle_step
                    # 使用旋转矩阵计算新位置
                    rot_direction = direction * np.cos(angle) + rotation_axis * np.sin(angle)
                    self.positions[i] = circle_center + radius * rot_direction
        
        # 计算声波参数
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
    def __init__(self, simulation, focus_point):
        self.simulation = simulation
        self.fig = plt.figure(figsize=(8,6))
        self.ax = plt.gca()
        self.im = plt.imshow(simulation.u_curr.T, cmap='coolwarm', origin='lower',
                            extent=[0, simulation.parameter.Lx, 0, simulation.parameter.Ly],
                            vmin=-0.5, vmax=0.5)
        plt.colorbar(label='Wave Amplitude')
        plt.title('Ultrasound Beamforming Simulation')
        
        # 获取聚焦点位置（保存为类属性以便在update中使用）
        self.center_element_idx = len(simulation.array_transducer.positions) // 2
        self.center_pos = simulation.array_transducer.positions[self.center_element_idx]
        # 设置聚焦点位置
        self.focus_point = focus_point  # 聚焦点位置
        
        # 创建图例句柄
        self.scatter_handle = None
        self.line_handle = None

    def update(self, frame):
        """更新动画"""
        self.simulation.update(frame)
        self.im.set_data(self.simulation.u_curr.T)
        
        # 移除之前的散点图和连接线（如果存在）
        if self.scatter_handle:
            self.scatter_handle.remove()
        if self.line_handle:
            self.line_handle.remove()
        
        # 重新绘制换能器阵元位置
        self.scatter_handle = self.ax.scatter(self.simulation.array_transducer.positions[:, 0],
                                             self.simulation.array_transducer.positions[:, 1],
                                             c='black', marker='o', s=1, label='Transducer Elements', zorder=10)
        
        # 重新绘制从中心阵元到聚焦点的连接线
        self.line_handle, = self.ax.plot([self.center_pos[0], self.focus_point[0]], 
                                        [self.center_pos[1], self.focus_point[1]], 
                                        'green', linewidth=3, label='Focus Line', zorder=10)
        
        # 更新图例
        self.ax.legend()
        
        return [self.im, self.scatter_handle, self.line_handle]

class Display_static:
    def __init__(self, simulation, focus_point, target_time):
        """初始化静态显示对象
        Args:
            simulation: BeamformingSimulator对象
            focus_point: 聚焦点坐标
            target_time: 目标时间点（单位：微秒）
        """
        self.simulation = simulation
        self.focus_point = focus_point
        
        # 计算需要更新的帧数
        target_frame = int(target_time * 1e-6 / simulation.dt)
        
        # 更新到目标时间点
        for frame in range(target_frame):
            simulation.update(frame)
        
        # 创建图形
        self.fig = plt.figure(figsize=(8,6))
        self.ax = plt.gca()
        
        # 显示波场
        self.im = plt.imshow(simulation.u_curr.T, cmap='coolwarm', origin='lower',
                            extent=[0, simulation.parameter.Lx, 0, simulation.parameter.Ly],
                            vmin=-0.5, vmax=0.5)
        plt.colorbar(label='Wave Amplitude')
        plt.title(f'Ultrasound Beamforming Static View (t = {target_time} μs)')
        
        # 获取中心阵元位置
        self.center_element_idx = len(simulation.array_transducer.positions) // 2
        self.center_pos = simulation.array_transducer.positions[self.center_element_idx]
        
        # 绘制换能器阵元位置
        self.ax.scatter(simulation.array_transducer.positions[:, 0],
                       simulation.array_transducer.positions[:, 1],
                       c='black', marker='o', s=1, label='Transducer Elements', zorder=10)
        
        # 绘制从中心阵元到聚焦点的连接线
        self.ax.plot([self.center_pos[0], focus_point[0]], 
                     [self.center_pos[1], focus_point[1]], 
                     'green', linewidth=3, label='Focus Line', zorder=10)
        
        # 显示图例
        self.ax.legend()
        
        # 显示网格
        self.ax.grid(True, linestyle='--', alpha=0.3)

