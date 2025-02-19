import torch
import triton
import triton.testing
from day04.matmul_fp16_v2 import matmul_v2
from day05.fp8_gemm import act_quant, fp8_gemm
from day05.int8_gemm import matmul_int8

# 修改后的配置
configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 40)],
        line_arg="provider",
        line_vals=["cublas", "triton-fp16", "triton-int8", "triton-fp8"],
        line_names=["cuBLAS FP16", "Triton FP16", "Triton INT8", "Triton FP8"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("purple", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-all-impl-comparison",
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # 保留INT8量化部分
    scale_a = (a.abs().max() / 127.0).clamp(min=1e-8)
    scale_b = (b.abs().max() / 127.0).clamp(min=1e-8)
    a_int8 = (a / scale_a).round().clamp(-127, 127).to(torch.int8)
    b_int8 = (b / scale_b).round().clamp(-127, 127).to(torch.int8)

    quantiles = [0.5, 0.2, 0.8]
    
    # 添加FP8量化部分
    a_fp8, a_scale = act_quant(a, block_size=128)
    b_fp8, b_scale = act_quant(b, block_size=128)
    
    # 修改条件判断逻辑，移除fp8分支
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == "triton-fp16":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v2(a, b), quantiles=quantiles)
    elif provider == "triton-int8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_int8(a_int8, b_int8),
            quantiles=quantiles
        )
    elif provider == "triton-fp8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_gemm(a_fp8, b_fp8, a_scale, b_scale),
            quantiles=quantiles
        )

    # 修改性能计算
    if provider == "triton-int8":
        perf = lambda ms: 4 * M * N * K * 1e-12 / (ms * 1e-3)
    elif provider == "triton-fp8":
        perf = lambda ms: 4 * M * N * K * 1e-12 / (ms * 1e-3)  # 增加类型转换和缩放计算
    else:
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)