import os
# os.environ['QT_LOGGING_RULES'] = 'qt.qpa.xcb.warning=false'
# os.environ['MPLBACKEND'] = 'Agg'  # 强制使用非交互式后端

import torch
import triton
import triton.testing
import matplotlib.pyplot as plt
from softmax_v1 import softmax_v1
from softmax_v2 import softmax_v2
import time
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置设备
DEVICE = torch.device('cuda')

# 定义用于 torch.compile 的 softmax 函数
@torch.compile
def compiled_softmax(x):
    # Step 1: 计算最大值（按行）
    max_val = torch.max(x,  dim=-1, keepdim=True).values  #* 读取 M*N，写入 M*1 
    # Step 2: 减去最大值（防止数值溢出）
    x_sub = x - max_val                                   #* 读取 M*N + M*1，写入 M*N 
    # Step 3: 指数运算 
    x_exp = torch.exp(x_sub)                               #* 读取 M*N，写入 M*N 
    # Step 4: 计算分母（按行的和）
    sum_exp = torch.sum(x_exp,  dim=-1, keepdim=True)      #* 读取 M*N，写入 M*1 
    # Step 5: 归一化 
    output = x_exp / sum_exp                              #* 读取 M*N + M*1，写入 M*N 
    return output 



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作x轴的参数名
        x_vals=[37984 * i for i in range(1, 5)],  # N的不同可能值
        line_arg='provider',  # 对应图中不同线条的参数名
        line_vals=['triton', 'torch.softmax', 'torch_compile'],  # line_arg的可能值
        line_names=[
            "Triton",
            "Torch",
            "Torch (Compiled)",
        ],  # 线条的标签名
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # 线条样式
        ylabel="GB/s",  # y轴标签名
        plot_name="softmax-performance",  # 图表名称，也用作保存图表的文件名
        args={'M': 4096},  # 不在x_names和y_name中的函数参数值
    ))
def benchmark(M, N, provider):
    # 创建输入数据
    x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
    
    # 创建CUDA流以确保准确的计时
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    
    # 根据provider选择不同的实现
    if provider == 'torch.softmax':
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_v2(x))
    elif provider == 'torch_compile':
        # 使用编译后的函数进行基准测试
        ms = triton.testing.do_bench(lambda: compiled_softmax(x))
    
    # 计算带宽 (GB/s)
    # 公式: 2 * 元素数量 * 每个元素大小 * 1e-9 / (毫秒 * 1e-3)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    # 运行基准测试
    benchmark.run(print_data=True, show_plots=True, save_path='./') 