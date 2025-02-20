import torch
import triton
import triton.language as tl
from triton import Config

@triton.jit
def act_quant_kernal(
    x_ptr,
    y_ptr,
    s_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """
    修正后的量化核函数
    :param x_ptr: 输入矩阵指针
    :param y_ptr: 输出矩阵指针
    :param s_ptr: 缩放系数指针
    :param BLOCK_SIZE: 块大小（编译时常量）
    """
    pid = tl.program_id(axis=0)
    # 计算当前block的偏移量
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    
    #* 计算block的最大绝对值
    max_val = tl.max(tl.abs(x), axis=0)
    s = max_val / 448.0  #! 量化系数是 float32的
    
    # 量化操作
    y = x / s  #! s的布局是 [... , hidden_dim / block_size]
    
    # 存储结果
    tl.store(y_ptr + offset , y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    修正后的量化函数，处理维度计算问题
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0

    # 创建输出张量（保持输入维度）
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # 调整缩放因子维度为 [..., K/block_size, 1]
    s_shape = (*x.size()[:-1], x.size(-1) // block_size, 1)
    s = x.new_empty(s_shape, dtype=torch.float32)
    
    #! 为了兼容多维度矩阵相乘
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    # 启动内核时添加显式维度参数
    act_quant_kernal[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# 示例配置（需要根据实际情况调整）
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr, b_s_ptr, 
    M,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 修正后的指针计算
    a_ptr = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptr = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    
    # 修正缩放因子索引
    a_s_ptr = a_s_ptr + (pid_m * k)
    b_s_ptr = b_s_ptr + (pid_n * k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        # 修正mask条件
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - i * BLOCK_SIZE_K)
        mask_b = (offs_k[:, None] < K - i * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_ptr, mask=mask_b, other=0.0)
        
        a_s = tl.load(a_s_ptr + i)
        b_s = tl.load(b_s_ptr + i)
        
        accumulator += tl.dot(a, b) * a_s * b_s
        
        # 修正指针递增
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=mask)


# # 修改自动调优配置（更适合FP8的配置）
# FP8_GEMM_CONFIGS = [
#     Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4),
#     Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4)
# ]

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    优化的FP8矩阵乘法内核
    - 使用分块策略提升内存访问效率
    - 支持自动调优不同块大小和流水线阶段
    - 包含动态缩放因子处理
    """
    # 计算当前块的起始位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算块的偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 分块矩阵乘法
    for k in range(0, K, BLOCK_SIZE_K):
        # 加载A矩阵块
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # 加载B矩阵块
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 加载缩放因子
        a_scale = tl.load(a_scale_ptr + pid_m)
        b_scale = tl.load(b_scale_ptr + pid_n)
        
        # 矩阵乘法累加
        acc += tl.dot(a.to(tl.float32) * a_scale, b.to(tl.float32) * b_scale)
    
    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



# def fp8_gemm(a: torch.Tensor, b: torch.Tensor):
#     """修改后的接口，接受FP16输入"""
#     # 内部量化处理
#     a_fp8, a_scale = act_quant(a)
#     b_fp8, b_scale = act_quant(b)
    
#     # 原有计算逻辑
#     M, K = a.shape
#     K, N = b.shape
#     c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
#     # 调整网格计算
    # grid = lambda meta: (
    #     triton.cdiv(M, meta['BLOCK_SIZE_M']),
    #     triton.cdiv(N, meta['BLOCK_SIZE_N'])
    # )
    
#     # 移除手动指定的BLOCK_SIZE参数
#     fp8_gemm_kernel[grid](
#         a_fp8, b_fp8, c,
#         a_scale, b_scale,
#         M, N, K  # 只传递必要参数
#     )
#     return c

def fp8_gemm(a: torch.Tensor, b: torch.Tensor):
    """
    FP8矩阵乘法接口函数
    参数:
        a: (M, K) 输入矩阵，FP8格式
        b: (K, N) 输入矩阵，FP8格式
        a_scale: (M,) A矩阵的缩放因子
        b_scale: (N,) B矩阵的缩放因子
    返回:
        c: (M, N) 输出矩阵，FP32格式
    """
    # 参数校验
    assert b.dtype == torch.float32, "输入矩阵必须是FP32，进行动态量化"
    assert a.is_contiguous() and b.is_contiguous(), "输入矩阵必须是连续内存布局"
    
    a_fp8, a_scale = act_quant(a)
    b_fp8, b_scale = act_quant(b)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )
    
    # 启动内核（添加缺失的参数）
    fp8_gemm_kernel[grid](
        a_fp8, b_fp8, c,
        a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),  # 添加步长参数
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,  # 添加默认块大小参数
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        num_stages=4  # 添加流水线阶段参数
    )
    return c




