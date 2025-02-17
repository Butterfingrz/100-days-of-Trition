import os
# os.environ['QT_LOGGING_RULES'] = 'qt.qpa.xcb.warning=false'
# os.environ['MPLBACKEND'] = 'Agg'  # 强制使用非交互式后端

import torch
import triton
import matplotlib.pyplot as plt
from softmax import softmax_2_rows
import time
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def benchmark_pytorch_softmax(x, num_warmup=25, num_tests=100):
    # 预热
    for _ in range(num_warmup):
        _ = torch.nn.functional.softmax(x, dim=1)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_tests):
        _ = torch.nn.functional.softmax(x, dim=1)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / num_tests

def benchmark_triton_softmax(x, num_warmup=25, num_tests=100):
    # 预热
    for _ in range(num_warmup):
        _ = softmax_2_rows(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_tests):
        _ = softmax_2_rows(x)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / num_tests

def run_benchmark():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    pytorch_times = []
    triton_times = []
    
    for size in sizes:
        # 创建输入数据
        x = torch.randn(size, size, device='cuda')
        
        # 测试 PyTorch 实现
        pytorch_time = benchmark_pytorch_softmax(x)
        pytorch_times.append(pytorch_time * 1000)  # 转换为毫秒
        
        # 测试 Triton 实现
        triton_time = benchmark_triton_softmax(x)
        triton_times.append(triton_time * 1000)  # 转换为毫秒
        
        print(f"Size: {size}")
        print(f"PyTorch time: {pytorch_time*1000:.3f} ms")
        print(f"Triton time: {triton_time*1000:.3f} ms")
        print(f"Speedup: {pytorch_time/triton_time:.2f}x")
        print("-" * 50)
    
    sizes = [i for i in range(len(pytorch_times))]
    plt.plot(sizes, pytorch_times, label='PyTorch')
    plt.plot(sizes, triton_times, label='Triton')
    plt.xlabel('input_size (N x N)')
    plt.ylabel('time (ms)')
    plt.title('PyTorch vs Triton Softmax Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('softmax_benchmark.png')


if __name__ == "__main__":
    run_benchmark() 