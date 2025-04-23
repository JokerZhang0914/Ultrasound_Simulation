# Ultrasound Simulation System

## Overview
Medical ultrasound simulation platform with three main functions:
- **BF_sim.py**: Ultrasound beamforming simulation with transducer arrays
- **sim1.py**: 1D ultrasound wave propagation simulation
- **sim2.py**: 2D ultrasound wave propagation simulation

## BF_sim
  - Dynamic display (Function 1): Real-time observation of sound field with selectable array types (linear/arc) and excitation modes (single/continuous)
  - Static display (Function 2): Sound field after propagation with arc array, selectable excitation modes (single/continuous)
  - BeamForming.py: Encapsulate the used Class
## sim1
  - Dynamic display (Function 1): Real-time observation of propagation process with echo reception
  - Echo observation (Function 2): Echo and envelope after propagation
## sim2
  - Dynamic display of ultrasound propagation in 2D
    - Multiple sound sources can be added
    - Rectangular, circular and elliptical regions can be added to simulate different media with different sound speeds

## Installation
```bash
pip install numpy matplotlib scipy
```

## Usage
```python
# Run beamforming simulation
python BeamForming/BF_sim.py

# Run basic simulation
python sim1.py

# Run advanced simulation
python sim2.py
```

---

# 超声仿真系统

## 概述
医学超声传播模拟包括三个功能
- **BF_sim.py**：换能器阵列实现超声波束合成
- **sim1.py**：一维超声波传播仿真
- **sim2.py**：二维超声波传播仿真

## BF_sim
- 含动态显示和静态显示
    - 动态显示（功能1）：动态观察声场，可选择阵列类型（线形/弧形）、发生情况(单次发生/连续发生)。
    - 静态显示（功能2）：得到传播一定时间后的声场（弧形阵列），可选择发生情况(单次发生/连续发生)。
- **BeamForming.py**：封装用到的类

## sim1
- 含动态显示和接收回波静态显示
    - 动态显示（功能1）：动态观察传播过程，同时看到回波的接收。
    - 观察回波（功能2）：得到传播一定时间后的回波及包络。

## sim2
- 动态显示二维情况下超声传播过程
    - 可添加多个声源
    - 可添加矩形、圆形、椭圆形区域，设置不同声速模拟不同介质
## 安装
```bash
pip install numpy matplotlib scipy
```

## 使用说明
```python
# 运行波束形成仿真
python BeamForming/BF_sim.py

# 运行基础仿真
python sim1.py

# 运行高级仿真
python sim2.py
```