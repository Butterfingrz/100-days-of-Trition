import torch
import triton
import triton.language as tl
from softmax import softmax_2_rows  # 从softmax.py导入我们实现的softmax函数
from softmax_v2 import softmax_2_rows_v2

# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义性能测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 作为x轴的参数名
        x_vals=[128 * i for i in range(2, 50)],  # x_name的不同可能值
        line_arg='provider',  # 对应图中不同线条的参数名
        line_vals=['triton', 'torch'],  # line_arg的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # 线条的标签名
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel="GB/s",  # y轴标签
        plot_name="softmax-performance",  # 图表名称，也用于保存图表
        args={'M': 4096},  # 不在x_names和y_name中的函数参数值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    #* 获取当前设备（GPU或CPU）的流（stream），用于异步计算
    stream = getattr(torch, DEVICE.type).Stream()
    #* 设置当前设备（GPU或CPU）的流（stream），用于异步计算
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_2_rows_v2(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)  # 这里添加了正确的缩进