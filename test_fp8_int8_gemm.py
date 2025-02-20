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
    
    # 统一使用FP16输入
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    a_fp32 = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b_fp32 = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # 修改后的FP8量化处理（在需要时动态量化）
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == "triton-fp16":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v2(a, b), quantiles=quantiles)
    elif provider == "triton-int8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_int8(a_fp32, b_fp32),
            quantiles=quantiles
        )
    elif provider == "triton-fp8":
        # 动态量化处理
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_gemm(a_fp32, b_fp32),  # 使用act_quant进行动态量化
            quantiles=quantiles
        )

    # 修改性能计算
    if provider in ["triton-int8", "triton-fp8"]:
        # 新公式：2*mnk + mk + kn
        total_flops = 2 * M * N * K + M * K + K * N
    else:
        # 原公式：2*mnk
        total_flops = 2 * M * N * K
        
    perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
        
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)