from BeamForming import Parameter, ArrayTransducer, Simulator, Display, Display_static
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # 创建参数对象
    parameter = Parameter(Lx=0.02, Ly=0.02, nx=200, ny=200)
    
    # 选择显示模式
    display_mode = input("请选择显示模式 (0为动态显示，1为静态显示): ") or '0'
    display_mode = '0' if display_mode not in ['0', '1'] else display_mode
    """
    TODO:模式3：波束扫描。实现聚焦点在一定y值上x值逐渐变化，得到声波传播相同且适当时间后的波长图，将这些波长图逐帧展示，实现波束扫描效果。
    难点：当前功能2设置显示时间后，计算所需时间过长。
    """
    # 选择阵列类型
    if display_mode == '1':
        array_type = 'curved'
    else:
        array_type = input("请选择阵列类型 (0为线性阵列，1为弧形阵列): ") or '0'
        array_type = 'curved' if array_type == '1' else 'linear'
    
    
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
    focus_point = np.array([0.01, 0.01])
    array_transducer.calculate_delays(focus_point)

    if display_mode == '0':
        # 动态显示模式
        emission_mode = input("请输入发射次数 (0为单次发射，1为连续发射): ") or '0'
        emission_mode = 'continuous' if emission_mode == '1' else '0'
        
        # 创建仿真对象
        simulation = Simulator(parameter, array_transducer, emission_mode=emission_mode)
        
        # 动态显示
        display = Display(simulation, focus_point)
        ani = FuncAnimation(display.fig, display.update,
                           frames=simulation.nt, interval=1, blit=True)
    else:
        # 静态显示模式
        # 创建仿真对象（静态显示只需单次发射）
        simulation = Simulator(parameter, array_transducer, emission_mode='continuous')
        
        target_time = float(input("请输入要显示的时间点（单位：微秒）: "))
        display = Display_static(simulation, focus_point, target_time)
    
    plt.show()

if __name__ == '__main__':
    main()