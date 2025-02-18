import torch
import triton
import triton.language as tl
from matmul_v1 import matmul_v1  # 导入我们实现的矩阵乘法函数

# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义性能测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # 修改为同时包含三个维度
        x_vals=[128 * i for i in range(2, 16)],  # 调整范围到32
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=[
            "Triton",
            "cublas",  # 保持cublas标签
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",  # 添加精度后缀
        args={},  # 移除固定参数，由x_vals动态传入
    ))
def benchmark(M, N, K, provider):
    # 生成fp16的输入矩阵
    x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    
    # 设置当前设备的流
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    # 定义分位数参数（新增）
    quantiles = [0.5, 0.2, 0.8]  # 中位数、20%分位数、80%分位数
    
    # 测量不同实现的运行时间（修改部分）
    if provider == 'torch':
        # 使用分位数参数获取多个时间指标
        ms, max_ms, min_ms = triton.testing.do_bench(
            lambda: torch.matmul(x, weight), 
            quantiles=quantiles
        )
    elif provider == 'triton':
        ms, max_ms, min_ms = triton.testing.do_bench(
            lambda: matmul_v1(x, weight),
            quantiles=quantiles
        )
    else:
        raise ValueError("Unknown provider")
    
    # 计算TFLOPS（修改返回值为三个指标）
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)  # 返回中位数、最大值、最小值

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True) 