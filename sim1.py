from turtle import end_fill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Parameter:
    def __init__(self, length, nx):
        self.length = length
        self.nx = nx
        self.dx = length / nx
        self.base_speed = 1500.0  # 基础声速 (m/s)
        self.regions = []  # 存储不同声阻抗区域的列表

    def add_region(self, start_pos, end_pos, ratio):
        """添加一个新的声阻抗区域"""
        self.regions.append({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'ratio': ratio
        })

    def get_speed(self):
        """生成声速分布数组"""
        c = np.full(self.nx, self.base_speed)
        for region in self.regions:
            start_idx = int(region['start_pos'] / self.dx)
            end_idx = int(region['end_pos'] / self.dx)
            c[start_idx:end_idx] = self.base_speed * region['ratio']
        return c

class Receiver:
    def __init__(self, nt, max_periods, receiver_pos, parameter):
        self.nt = nt
        self.max_periods = max_periods
        self.parameter = parameter
        # 将实际位置转换为索引位置
        self.receiver_pos = int(receiver_pos / self.parameter.dx)
        self.received_signal = np.zeros(nt * max_periods)
        self.current_period = 0
        self.total_frame_count = 0
        self.envelope = None

    def record_signal(self, u_curr, current_frame):
        """记录接收点的信号"""
        signal_index = self.current_period * self.nt + current_frame
        self.received_signal[signal_index] = u_curr[self.receiver_pos]

    def get_envelope(self):
        """使用Hilbert变换计算信号包络"""
        from scipy.signal import hilbert
        analytic_signal = hilbert(self.received_signal)
        self.envelope = np.abs(analytic_signal)
        return self.envelope

    def analyze_impedance(self):
        """分析声阻抗变化"""
        if self.envelope is None:
            self.get_envelope()
        
        # 找到包络的峰值
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self.envelope, height=0.1)
        
        # 计算峰值对应的时间和位置
        time_axis = np.linspace(0, len(self.received_signal) * self.parameter.dx / self.parameter.base_speed, len(self.received_signal))
        peak_times = time_axis[peaks]
        peak_positions = peak_times * self.parameter.base_speed
        peak_amplitudes = self.envelope[peaks]
        
        return peak_positions, peak_amplitudes

class Simulator:
    def __init__(self, parameter):
        # 仿真参数
        self.parameter = parameter
        self.f0 = 2.5e6  # 中心频率 (Hz)
        self.cycles = 4  # 周期数
        self.t_end = 25e-6  # 总时间 (s)
        self.CFL = 0.5  # CFL条件系数
        
        # 初始化声速场
        self.c = parameter.get_speed()
        
        # 计算时间步长
        self.dt = self.CFL * self.parameter.dx / np.max(self.c)
        self.nt = int(self.t_end / self.dt)
        
        # 初始化波场
        self.u_curr = np.zeros(self.parameter.nx)
        self.u_prev = np.zeros(self.parameter.nx)
        self.u_next = np.zeros(self.parameter.nx)
        
        # 生成声源
        self.source_active = True
        self.source = self._generate_source()
        
        # 边界条件
        self.alpha = 0.0

    def _generate_source(self):
        """生成衰减正弦波声源"""
        num_source_points = int(round(self.cycles/(self.f0*self.dt)))
        t_source = np.linspace(0, self.cycles/self.f0, num_source_points)
        return np.exp(-(t_source - 0.5*t_source[-1])**2/((t_source[-1]/4)**2)) * \
               np.sin(2*np.pi*self.f0*t_source)

    def update(self, current_frame):
        """更新波场"""
        # 更新波场内部节点
        self.u_next[1:-1] = (2*self.u_curr[1:-1] - self.u_prev[1:-1] + 
                            (self.dt**2/self.parameter.dx**2) * 
                            self.c[1:-1]**2 * 
                            (self.u_curr[2:] - 2*self.u_curr[1:-1] + self.u_curr[:-2]))
        
        # 应用吸收边界条件
        self.u_next[0] = self.u_curr[1]
        self.u_next[-1] = self.alpha * self.u_curr[-2]
        
        # 施加声源
        if current_frame < len(self.source) and self.source_active:
            self.u_next[0] = self.source[current_frame]
        
        if current_frame == len(self.source) - 1:
            self.source_active = False

        # 更新波场
        self.u_prev[:] = self.u_curr
        self.u_curr[:] = self.u_next

class Display:
    def __init__(self, simulation, receiver, mode=1):
        self.simulation = simulation
        self.receiver = receiver
        self.mode = mode
        if mode == 1:
            self.setup_plots_mode1()
        else:
            self.setup_plots_mode2()

    def setup_plots_mode1(self):
        """设置模式1的绘图对象（三图显示）"""
        self.fig, (self.ax2, self.ax1, self.ax3) = plt.subplots(3, 1, 
            height_ratios=[0.8, 3, 1], figsize=(10, 12))
        
        # 声速分布图
        c_ratio = self.simulation.c / self.simulation.parameter.base_speed
        self.ax2.plot(np.linspace(0, self.simulation.parameter.length, 
                                 self.simulation.parameter.nx), 
                     c_ratio, 'r-', label='Sound Speed Ratio')
        self.ax2.set_xlabel('Position (m)')
        self.ax2.set_ylabel('Speed Ratio (c/c0)')
        self.ax2.grid(True)
        self.ax2.legend()

        # 波场图
        self.line, = self.ax1.plot(np.linspace(0, self.simulation.parameter.length, 
                                               self.simulation.parameter.nx), 
                                  self.simulation.u_curr)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.set_xlabel('Position (m)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Ultrasound Wave Propagation with Impedance Change')

        # 接收信号图
        self.line_received, = self.ax3.plot([], [], 'g-', label='Received Signal')
        self.line_envelope, = self.ax3.plot([], [], 'r--', label='Signal Envelope')
        self.ax3.set_xlabel('Time (μs)')
        self.ax3.set_ylabel('Amplitude')
        self.ax3.set_title('Received Signal and Envelope')
        self.ax3.grid(True)
        self.ax3.legend()

        plt.tight_layout()

    def update(self, frame):
        """更新动画"""
        self.receiver.total_frame_count += 1
        current_frame = self.receiver.total_frame_count % self.simulation.nt
        self.receiver.current_period = (self.receiver.total_frame_count // 
                                              self.simulation.nt) % self.receiver.max_periods

        # 更新波场
        self.simulation.update(current_frame)
        
        # 记录接收信号
        self.receiver.record_signal(self.simulation.u_curr, current_frame)
        
        # 更新图形
        self.line.set_ydata(self.simulation.u_curr)
        
        # 更新接收信号图
        valid_frames = min(self.receiver.total_frame_count, 
                         self.receiver.max_periods * self.simulation.nt)
        time_axis = np.linspace(0, valid_frames*self.simulation.dt*1e6, valid_frames)
        self.line_received.set_data(time_axis, 
                                   self.receiver.received_signal[:valid_frames])
        
        # 更新包络
        envelope = self.receiver.get_envelope()
        self.line_envelope.set_data(time_axis, envelope[:valid_frames])
        
        # 分析声阻抗变化
        if self.receiver.total_frame_count % self.simulation.nt == 0:
            positions, amplitudes = self.receiver.analyze_impedance()
            print("\n声阻抗变化位置 (m):", positions)
            print("相对幅度:", amplitudes)
        
        self.ax3.set_xlim(0, self.receiver.max_periods * self.simulation.t_end*1e6)
        self.ax3.set_ylim(-1.5, 1.5)
        
        return self.line, self.line_received, self.line_envelope

    def setup_plots_mode2(self):
        """设置模式2的绘图对象（接收信号波形）"""
        # 运行仿真直到60us
        end_time = int(input("请输入要显示的时间（单位：μs）：")) * 1e-6
        frames = int(end_time / self.simulation.dt)
        for frame in range(frames):
            self.simulation.update(frame)
            self.receiver.record_signal(self.simulation.u_curr, frame)
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        time_axis = np.linspace(0, frames*self.simulation.dt*1e6, frames)
        
        # 绘制接收信号
        self.ax.plot(time_axis, self.receiver.received_signal[:frames], 'g-', label='Received Signal')
        
        # 计算并绘制包络
        envelope = self.receiver.get_envelope()
        self.ax.plot(time_axis, envelope[:frames], 'r--', label='Envelope')
        
        # 标注峰值
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(envelope[:frames], height=0.1)
        peak_times = time_axis[peaks]
        peak_values = envelope[peaks]
        
        for t, v in zip(peak_times, peak_values):
            self.ax.annotate(f'{v:.2f}', xy=(t, v), xytext=(0, 10),
                            textcoords='offset points', ha='center')
        
        self.ax.set_xlabel('Time (μs)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Received Signal at 60μs with Envelope')
        self.ax.grid(True)
        self.ax.legend()
        plt.tight_layout()

def main():
    # 创建声阻抗分布对象
    parameter = Parameter(length=0.05, nx=1500)
    parameter.add_region(start_pos=0.01, end_pos=0.02, ratio=0.6)

    # 创建仿真对象
    simulation = Simulator(parameter)

    # 创建接收器
    receiver = Receiver(simulation.nt, max_periods=5, receiver_pos=0.0, parameter=parameter)

    # 选择显示模式（1：三图动态显示，2：接收信号波形）
    mode = int(input("请选择显示模式（1：三图动态显示，2：接收信号波形）："))
    
    # 创建可视化对象
    display = Display(simulation, receiver, mode)

    if mode == 1:
        # 创建动画
        ani = FuncAnimation(display.fig, display.update, 
                           frames=simulation.nt, interval=0.5, 
                           blit=True, repeat=True)
    
    plt.show()

if __name__ == '__main__':
    main()