from BeamForming import Parameter, ArrayTransducer, BeamformingSimulator, Display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # 创建参数对象
    parameter = Parameter(Lx=0.02, Ly=0.02, nx=200, ny=200)
    
    # 选择阵列类型
    array_type = input("请选择阵列类型 (0为线性阵列，1为弧形阵列): ") or '0'
    array_type = 'linear' if array_type == '0' else 'curved'
    
    # 创建换能器阵列
    array_transducer = ArrayTransducer(
        num_elements=16,  # 16个阵元
        element_pitch=0.0002,  # 阵元间距0.2mm
        element_width=0.0001,  # 阵元宽度0.1mm
        Lx=parameter.Lx,  # 仿真区域宽度
        Ly=parameter.Ly,  # 仿真区域高度
        array_type=array_type,  # 阵列类型
        radius=0.015 if array_type == 'curved' else None  # 弧形阵列半径15mm
    )
    
    # 设置聚焦点
    focus_point = np.array([0.01, 0.0025])
    array_transducer.calculate_delays(focus_point)

    # 选择发射模式 ('single'为单次发射，'continuous'为连续发射)
    emission_mode = input("请选择发射模式 (0为单次发射，1为连续发射): ") or '0'
    emission_mode = 'single' if emission_mode == '0' else 'continuous'

    # 创建仿真对象
    simulation = BeamformingSimulator(parameter, array_transducer, emission_mode=emission_mode)

    # 创建显示对象
    display = Display(simulation, focus_point)

    # 创建动画
    ani = FuncAnimation(display.fig, display.update,
                       frames=simulation.nt, interval=1, blit=True)
    plt.show()

if __name__ == '__main__':
    main()